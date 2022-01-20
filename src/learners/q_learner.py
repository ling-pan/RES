import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def softmax_weighting(self, q_vals):
        assert q_vals.shape[-1] != 1
        
        max_q_vals = th.max(q_vals, -1, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(self.args.res_beta * norm_q_vals)

        numerators = e_beta_normQ 
        denominators = th.sum(e_beta_normQ, -1, keepdim=True)

        softmax_weightings = numerators / denominators
        
        return softmax_weightings

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

            if self.args.res:
            # if False:
                assert self.args.double_q

                all_counterfactual_actions_qvals = []
                all_counterfactual_actions_target_qvals = []
                for agent_idx in range(cur_max_actions.shape[2]):
                    base_actions = copy.deepcopy(cur_max_actions)
                    # total_batch_size, num_agents
                    base_actions = base_actions.squeeze(-1).reshape(-1, cur_max_actions.shape[2])

                    # num_actions, 1
                    all_actions_for_an_agent = th.tensor([action_idx for action_idx in range(self.args.n_actions)]).unsqueeze(0)
                    # num_actions, total_batch_size: [[0, ..., 0], [1, ..., 1], ..., [4, ..., 4]]
                    all_actions_for_an_agent = all_actions_for_an_agent.repeat(base_actions.shape[0], 1).transpose(1, 0)
                    # formate to a column vector: total_batch_size x num_actions: [0, ..., 0, ...., 4, ..., 4]
                    all_actions_for_an_agent = all_actions_for_an_agent.reshape(-1, 1).squeeze()
                    
                    # total_batch_size x num_agents, num_actions (repeat the actions for num_actions times)
                    counterfactual_actions = base_actions.repeat(self.args.n_actions, 1).reshape(-1, base_actions.shape[1])

                    counterfactual_actions[:, agent_idx] = all_actions_for_an_agent

                    counterfactual_actions_qvals, counterfactual_actions_target_qvals = [], []
                    for action_idx in range(self.args.n_actions):
                        curr_counterfactual_actions = counterfactual_actions[action_idx * base_actions.shape[0] : (action_idx + 1) * base_actions.shape[0]]
                        curr_counterfactual_actions = curr_counterfactual_actions.reshape(cur_max_actions.shape[0], cur_max_actions.shape[1], cur_max_actions.shape[2], -1)

                        # batch_size, episode_len, num_agents
                        curr_counterfactual_actions_qvals = th.gather(mac_out_detach[:, 1:], dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim
                        curr_counterfactual_actions_target_qvals = th.gather(target_mac_out, dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim

                        # batch_size, episode_len, 1
                        curr_counterfactual_actions_qvals = self.mixer(
                            curr_counterfactual_actions_qvals, batch["state"][:, 1:]
                        )
                        curr_counterfactual_actions_qvals = curr_counterfactual_actions_qvals.reshape(
                            curr_counterfactual_actions_qvals.shape[0] * curr_counterfactual_actions_qvals.shape[1], 1
                        )
                        
                        curr_counterfactual_actions_target_qvals = self.target_mixer(
                            curr_counterfactual_actions_target_qvals, batch["state"][:, 1:]
                        )
                        curr_counterfactual_actions_target_qvals = curr_counterfactual_actions_target_qvals.reshape(
                            curr_counterfactual_actions_target_qvals.shape[0] * curr_counterfactual_actions_target_qvals.shape[1], 1
                        )

                        counterfactual_actions_qvals.append(curr_counterfactual_actions_qvals)
                        counterfactual_actions_target_qvals.append(curr_counterfactual_actions_target_qvals)

                    # batch_size x episode_len, num_actions
                    counterfactual_actions_qvals = th.cat(counterfactual_actions_qvals, 1)
                    counterfactual_actions_target_qvals = th.cat(counterfactual_actions_target_qvals, 1)

                    all_counterfactual_actions_qvals.append(counterfactual_actions_qvals)
                    all_counterfactual_actions_target_qvals.append(counterfactual_actions_target_qvals)
                
                # total_batch_size, num_agents, num_actions
                all_counterfactual_actions_qvals = th.stack(all_counterfactual_actions_qvals).permute(1, 0, 2)
                all_counterfactual_actions_target_qvals = th.stack(all_counterfactual_actions_target_qvals).permute(1, 0, 2)
                
                # total_batch_size, num_agents x num_actions
                all_counterfactual_actions_qvals = all_counterfactual_actions_qvals.reshape(all_counterfactual_actions_qvals.shape[0], -1)
                all_counterfactual_actions_target_qvals = all_counterfactual_actions_target_qvals.reshape(all_counterfactual_actions_target_qvals.shape[0], -1)

                softmax_weightings = self.softmax_weighting(all_counterfactual_actions_qvals)
                softmax_qtots = softmax_weightings * all_counterfactual_actions_target_qvals
                softmax_qtots = th.sum(softmax_qtots, 1, keepdim=True)

                softmax_qtots = softmax_qtots.reshape(rewards.shape[0], rewards.shape[1], rewards.shape[2])
                target_max_qvals = softmax_qtots
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.res:
            future_episode_return = batch["future_discounted_return"][:, :-1]
            q_return_diff = (chosen_action_qvals - future_episode_return.detach())

            v_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()

            loss += self.args.res_lambda * v_l2
            
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
