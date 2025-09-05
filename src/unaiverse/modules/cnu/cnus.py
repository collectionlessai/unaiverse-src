"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import math
import torch
from .psi import psi
import torch.nn.functional as F


class CNUs(torch.nn.Module):

    def __init__(self, q=1, d=2, m=3, u=4, delta=3,
                 gamma_alpha=0.1, tau_alpha=0.5, tau_mu=100, tau_eta=100,
                 upd_m="WTA", upd_k="ad_hoc_WTA",
                 beta_k=0.001,
                 psi_fn="identity",
                 scramble=False):
        """
        :param q: number of neurons
        :param d: size of each key
        :param m: number of keys/memory units
        :param u: size of each memory unit
        :param gamma_alpha: softmax temperature (key matching)
        :param tau_alpha: threshold on the attention score of the winning key, to eventually trigger scrambling
        :param tau_mu: number of steps below which a key is considered to be not-used enough
        :param tau_eta: number of steps after which a key is considered old
        :param delta: number of top attention responses to select (top-delta)
        :param upd_m: update memory strategy (None, 'WTA')
        :param upd_k: update key strategy (None, 'ad_hoc_WTA', 'grad_WTA')
        :param beta_k: learning rate for key-update purposes when upd_k is 'ad_hoc_WTA'
        :param psi_fn: function to project the neuron input onto the key space
        :param scramble: triggers the key/memory scrambling routine when upd_k is 'ad_hoc_WTA'
        """

        super(CNUs, self).__init__()
        assert upd_m in (None, 'WTA'), "Unknown value for upd_m, it must be None or 'WTA'"
        assert upd_k in (None, 'ad_hoc_WTA', 'grad_WTA'), "Unknown value for upd_k, it must be " \
                                                          "None, 'ad_hoc_WTA', or 'grad_WTA'"
        assert upd_m is None or (upd_m == 'WTA' and upd_k is not None), \
            "If upd_m is 'WTA', then upd_k must be ad_hoc_WTA or grad_WTA (it cannot be None)"
        self.q = q
        self.d = d
        self.m = m
        self.u = u
        self.gamma_alpha = gamma_alpha
        self.tau_alpha = tau_alpha
        self.tau_mu = tau_mu
        self.tau_eta = tau_eta
        self.scramble = scramble
        self.delta = min(delta, self.m)
        self.upd_m = upd_m
        self.upd_k = upd_k
        self.beta_k = beta_k
        self.psi_fn = psi_fn
        self.debug = False  # Temporarily used
        self.reset_memories = True

        # Creating keys (self.K) and memories (self.M)
        self.M = torch.nn.Parameter(torch.empty((self.q, self.m, self.u), dtype=torch.float32))
        if self.upd_k == "ad_hoc_WTA":
            self.register_buffer('K', torch.zeros((self.q, self.m, self.d)))
        else:
            self.K = torch.nn.Parameter(torch.empty((self.q, self.m, self.d), dtype=torch.float32))

        # Buffers for ad_hoc_WTA key updates (average usefulness register buffer "mu" and age "eta")
        if self.upd_k == "ad_hoc_WTA":
            self.register_buffer('mu', torch.zeros(self.q, m, dtype=torch.float))
            self.register_buffer('eta', torch.ones((self.q, m), dtype=torch.float) * self.tau_eta)
            if self.debug:
                self.register_buffer('key_counter', torch.zeros(self.q, m, dtype=torch.float))
        else:
            self.mu = None
            self.eta = None

        # Scrambling stats
        self.register_buffer('scrambling_count', torch.zeros(self.q, dtype=torch.long))

        # Initializing memories and keys
        self.reset_parameters()

    def reset_parameters(self):
        self.__reset_keys()
        if self.reset_memories:
            self.__reset_memories()
        self.__reset_counters()

    def compute_weights(self, x):

        # Shortcuts (notice that "self.delta" is called "k" in shortcuts, while "self.delta-1" is called "z")
        q, m, u, d, k = self.q, self.m, self.u, self.d, self.delta
        b = x.shape[0]
        M_qmu = self.M

        # Ensuring keys are normalized (not needed with ad_hoc_WTA updates)
        if self.upd_k != 'ad_hoc_WTA':
            self.__normalize_keys()
        else:
            x = x.detach()  # In ad-hoc WTA, no gradient is propagated to the layers below through key-matching

        # Mapping the input to the key space using the psi function
        x_bd = psi(x, self.psi_fn, key_size=d, normalize=True)

        # Finding the top responses and indices for the attention procedure
        top_responses_bqk, top_indices_bqk = self.__top_k_attention(x_bd)

        # Probabilities
        top_alpha_bqk = torch.softmax((self.gamma_alpha / math.sqrt(d)) * top_responses_bqk, dim=2)

        if self.debug:

            # Getting the top-1 indices for the current mini-batch
            top1_indices_qb = top_indices_bqk[..., 0].t()
            self.key_counter.data.scatter_add_(dim=1,
                                               index=top1_indices_qb,
                                               src=torch.ones_like(top1_indices_qb, dtype=self.key_counter.dtype))

        # Updating keys with the ad-hoc scheme (also refreshing top-stuff: responses, indices, alpha)
        if self.training and self.upd_k == 'ad_hoc_WTA':
            top_responses_bqk, top_indices_bqk, top_alpha_bqk = \
                self.__update_keys_and_counters(x_bd, top_responses_bqk, top_indices_bqk, top_alpha_bqk)

        # Reading memories and blending them
        if self.upd_m is None:

            # Preparing to read memory units and to blend them
            M_exp_bqmu = M_qmu.view(1, q, m, u).expand(b, q, m, u)

            # Getting top memory units
            top_M_bqku = torch.gather(M_exp_bqmu, dim=2,
                                      index=top_indices_bqk.view(b, q, k, 1).expand(b, q, k, u))

            # Mixing memory units by attention scores
            # -> top_alpha_bqk: [b,q,k], that we un-squeeze to [b,q,1,k]
            # -> top_M_bqku: [b,q,k,u]
            # -> W_bqu: matmul([(b,q),1,k], [(b,q),k,u]) = [b,q,1,u] that we squeeze to [b,q,u]
            W_bqu = torch.matmul(top_alpha_bqk.view(b, q, 1, k), top_M_bqku).squeeze(2)

        elif self.upd_m == 'WTA':

            # Preparing to read memory units and to blend them
            M_exp_bqmu = M_qmu.view(1, q, m, u).expand(b, q, m, u)

            # Dealing with top-1 stuff
            top1_M_exp_bq1u = torch.gather(M_exp_bqmu, dim=2,
                                           index=top_indices_bqk[..., 0:1].view(b, q, 1, 1).expand(b, q, 1, u))

            # Mixing memory units by attention scores
            # -> top1_alpha_bqk: [b,q,k], that we select to [b,k,1] un-squeeze to [b,q,1,1]
            # -> top1_M_exp_bq1u: [b,q,1,u]
            # -> W_bqu: [b,q,1,1] * [b,q,1,u] = [b,q,1,u] that we squeeze to [b,q,u]
            top1_W_bqu = (top_alpha_bqk[..., 0:1].view(b, q, 1, 1) * top1_M_exp_bq1u).squeeze(2)

            # Dealing with top-2-and-following stuff
            top2on_M_exp_bqzu = torch.gather(M_exp_bqmu.detach(), dim=2,
                                             index=top_indices_bqk[..., 1:].view(b, q, k-1, 1).expand(b, q, k-1, u))
            top2on_alpha_bqz = top_alpha_bqk[:, :, 1:]
            if self.upd_k == 'grad_WTA':
                top2on_alpha_bqz = top2on_alpha_bqz.detach()

            # Mixing memory units by attention scores
            # -> top2on_alpha_bqz: [b,q,k-1], that we un-squeeze to [b,q,1,k-1]
            # -> top2on_M_exp_bqzu: [b,q,k-1,u]
            # -> W_bqu: matmul([(b,q),1,k-1], [(b,q),k-1,u]) = [b,q,1,u] that we squeeze to [b,q,u]
            top2on_W_bqu = torch.matmul(top2on_alpha_bqz.view(b, q, 1, k-1), top2on_M_exp_bqzu).squeeze(2)

            # Merging top1 and top-2-and-following stuff
            W_bqu = top1_W_bqu + top2on_W_bqu

        else:

            # What is going on?
            raise NotImplementedError

        return W_bqu

    def forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def __reset_memories(self):
        bound = 1. / math.sqrt(self.u)
        torch.nn.init.uniform_(self.M, -bound, bound)

    @torch.no_grad()
    def __reset_keys(self):
        bound = 1. / math.sqrt(self.d)
        torch.nn.init.uniform_(self.K, -bound, bound)
        self.K.data = F.normalize(self.K, p=2.0, dim=2, eps=1e-12, out=None)

    def __reset_counters(self):
        if self.mu is not None:
            self.mu = torch.zeros_like(self.mu)
        if self.eta is not None:
            self.eta.fill_(self.tau_eta)

    def reset_counter(self):
        if self.debug:
            self.key_counter.data = torch.zeros_like(self.key_counter)

    def __top_k_attention(self, x_bd):

        # Matmul([b,d], [d,qm]) = [b,qm], then reshaped (view) to [b,q,m]
        responses_bqm = torch.matmul(x_bd, self.K.view(self.q * self.m, self.d).t()).view(-1, self.q, self.m)
        top_responses_bqk, top_indices_bqk = torch.topk(responses_bqm, k=self.delta, dim=2, largest=True, sorted=True)
        return top_responses_bqk, top_indices_bqk

    @torch.no_grad()
    def __normalize_keys(self, ids=None):
        """
        :param ids: None or a vector with "self.q" elements, with the indices of the keys to consider for each neuron
        """
        if ids is not None:
            ids_exp_q1d = ids.view(self.q, 1, 1).expand(self.q, 1, self.d)
            keys_q1d = torch.gather(self.K, dim=1, index=ids_exp_q1d)
            keys_q1d = F.normalize(keys_q1d, p=2.0, dim=2, eps=1e-12, out=None)
            self.K.scatter_(dim=1, index=ids_exp_q1d, src=keys_q1d)
            return keys_q1d
        else:
            self.K.data = F.normalize(self.K, p=2.0, dim=2, eps=1e-12, out=None)
            return self.K

    def __update_keys_and_counters(self, x_bd, top_responses_bqk, top_indices_bqk, top_alphas_bqk):

        # Saving some shortcuts, notice that "self.delta" is called "k" here
        b, q, d, m, k, u = x_bd.shape[0], self.q, self.d, self.m, self.delta, self.u
        K_qmd = self.K
        mu_qm, eta_qm = self.mu, self.eta

        # Getting the top-1 indices for the current mini-batch
        top1_indices_qb = top_indices_bqk[..., 0].t()

        # Temporarily used
        if self.debug:
            K_initial_qmd = self.K.clone()
            M_initial_qmd = self.M.clone()
            mu_initial_qm = self.mu.clone()
            eta_initial_qm = self.eta.clone()
            top1_indices_initial_qb = top1_indices_qb.clone()
        else:
            K_initial_qmd, M_initial_qmd, mu_initial_qm = None, None, None
            eta_initial_qm, top1_indices_initial_qb = None, None

        # Determining if we need to scramble keys and memories (if scrambling is enabled)
        # (up to one key/memory per neuron, even when the batch size is greater than one)
        if self.scramble:

            # Computing a boolean mask that tells what neurons should be subject to scrambling (scramble_q),
            # a boolean mask associated with the weak keys,
            # and the indices of the elements of x_bd (batch elements) that should replace the
            # scrambled keys (weak_batch_elements_q)
            scramble_q, weak_keys_mask_qm, weak_batch_elements_q = \
                self.__evaluate_scrambling_conditions(top_responses_bqk)

            # Temporarily used
            if self.debug:
                self.__debug_pre_scrambling(top_indices_bqk, top_alphas_bqk, scramble_q,
                                            weak_keys_mask_qm, weak_batch_elements_q, x_bd)

            # If at least one neuron requires a scrambling operation, we do scramble! (in-place)
            if torch.any(scramble_q):
                self.__scramble(x_bd, scramble_q, weak_keys_mask_qm, weak_batch_elements_q, top1_indices_qb)

            # Temporarily used
            if self.debug:
                self.__debug_changes_with_respect_to(K_initial_qmd, M_initial_qmd, mu_initial_qm, eta_initial_qm,
                                                     top1_indices_initial_qb, top1_indices_qb,
                                                     msg="Right after scrambling...")

        # Computing variations to apply to winning keys, eventually using an adaptive learning rate
        key_variations_qbd = (self.beta_k * x_bd).view(1, b, d).expand(q, b, d)  # Delta to add to each winning key

        # Updating the winning keys (one winning key per neuron, for each batch example)
        K_qmd.scatter_add_(dim=1,
                           index=top1_indices_qb.view(q, b, 1).expand(q, b, d),
                           src=key_variations_qbd)  # Adding variations to winning keys

        # Recomputing or updating responses (in the case of batch size > 1, we recompute them all)
        top_responses_bqk, top_indices_bqk, top1_indices_qb = \
            self.__update_top_k_attention(x_bd, top_responses_bqk, top_indices_bqk, top1_indices_qb)

        # Recomputing the softmax over the (now updated) top responses
        top_alphas_bqk = torch.softmax((self.gamma_alpha / math.sqrt(d)) * top_responses_bqk, dim=2)  # [b,q,k]

        # Resetting ages of winning keys
        eta_qm.scatter_(1, top1_indices_qb, 0.)

        # Updating counters: usages ("mu") for the winning keys and ages ("eta") for all the keys
        mu_qm.scatter_add_(dim=1,
                           index=top1_indices_qb,
                           src=torch.ones_like(top1_indices_qb, dtype=mu_qm.dtype))  # Winning keys
        eta_qm += b  # All the keys

        # Temporarily used
        if self.debug:
            self.__debug_changes_with_respect_to(K_initial_qmd, M_initial_qmd, mu_initial_qm, eta_initial_qm,
                                                 top1_indices_initial_qb, top1_indices_qb,
                                                 msg="At the end of the whole key-and-counters update procedure...")

        return top_responses_bqk, top_indices_bqk, top_alphas_bqk

    def __evaluate_scrambling_conditions(self, top_responses_bqk):

        # Shortcuts
        mu_qm = self.mu
        eta_qm = self.eta

        # Computing the max of alphas (we take the smallest of them in case of batch sizes greater than one)
        top1_responses_bq = top_responses_bqk[..., 0]  # Max of responses
        max_of_responses_q, weak_batch_elements_q = torch.min(top1_responses_bq, dim=0)

        # Finding the weak keys, if any (boolean mask)
        weak_keys_mask_qm = torch.logical_and(mu_qm < self.tau_mu, eta_qm >= self.tau_eta)

        # Determining on what neurons scrambling should be applied (boolean mask)
        scramble_q = torch.logical_and(max_of_responses_q < self.tau_alpha, torch.any(weak_keys_mask_qm, dim=1))

        return scramble_q, weak_keys_mask_qm, weak_batch_elements_q

    def __scramble(self, x_bd, scramble_q, weak_keys_mask_qm, weak_batch_elements_q, top1_indices_qb):

        # Shortcuts
        q, d, u = self.q, self.d, self.u
        K_qmd, M_qmu, mu_qm, eta_qm = self.K, self.M, self.mu, self.eta

        # Stats
        self.scrambling_count[scramble_q] += 1

        # Finding the indices of the candidate keys to scramble (one per neuron)
        scramble_candidates_keys_q = torch.max(eta_qm * weak_keys_mask_qm.to(torch.float), dim=1)[1]

        # Neurons that must and must-not be subject to scrambling operations
        scramble_q = scramble_q.to(torch.float)
        no_scramble_q = torch.logical_not(scramble_q).to(torch.float)  # Boolean mask

        # Scrambling keys
        scramble_candidates_keys_exp_q1d = scramble_candidates_keys_q.view(q, 1, 1).expand(q, 1, d)
        scramble_q11 = scramble_q.view(q, 1, 1)
        no_scramble_q11 = no_scramble_q.view(q, 1, 1)

        new_keys_q1d = x_bd.gather(dim=0, index=weak_batch_elements_q.view(q, 1).expand(q, d)).view(q, 1, d)
        old_keys_q1d = K_qmd.gather(dim=1, index=scramble_candidates_keys_exp_q1d)

        K_qmd.scatter_(dim=1,
                       index=scramble_candidates_keys_exp_q1d,
                       src=new_keys_q1d * scramble_q11 + old_keys_q1d * no_scramble_q11)

        # Scrambling memories
        with torch.no_grad():
            scramble_candidates_keys_exp_q1u = scramble_candidates_keys_q.view(q, 1, 1).expand(q, 1, u)
            weak_keys_q1 = top1_indices_qb.gather(dim=1, index=weak_batch_elements_q.view(q, 1))
            weak_keys_exp_q1u = weak_keys_q1.view(q, 1, 1).expand(q, 1, u)

            new_memories_q1u = M_qmu.gather(dim=1, index=weak_keys_exp_q1u)
            old_memories_q1u = M_qmu.gather(dim=1, index=scramble_candidates_keys_exp_q1u)

            M_qmu.scatter_(
                dim=1,
                index=scramble_candidates_keys_exp_q1u,
                src=new_memories_q1u * scramble_q11 + old_memories_q1u * no_scramble_q11)

        # Updating indices of the winning keys (in-place!)
        weak_batch_elements_q1 = weak_batch_elements_q.view(q, 1)
        scramble_q1 = scramble_q.view(q, 1)
        no_scramble_q1 = no_scramble_q.view(q, 1)

        new_top1_indices_q1 = scramble_candidates_keys_q.view(q, 1)
        old_top1_indices_q1 = top1_indices_qb.gather(dim=1, index=weak_batch_elements_q1)

        top1_indices_qb.scatter_(
            dim=1, index=weak_batch_elements_q1,
            src=(new_top1_indices_q1 * scramble_q1 + old_top1_indices_q1 * no_scramble_q1).to(torch.long))

        # Resetting to zero the usage counts ("mu") for keys that were scrambled
        scramble_candidates_keys_q1 = scramble_candidates_keys_q.view(q, 1)

        new_values = 0.
        old_values_q1 = mu_qm.gather(dim=1, index=scramble_candidates_keys_q1)

        mu_qm.scatter_(
            dim=1,
            index=scramble_candidates_keys_q1,
            src=new_values * scramble_q1 + old_values_q1 * no_scramble_q1)

    def __update_top_k_attention(self, x_bd, top_responses_bqk, top_indices_bqk, top1_indices_qb):
        q, d, b = self.q, self.d, x_bd.shape[0]

        if b == 1:

            # Normalizing the winning keys (recall that b=1 here)
            normalized_winning_keys_q1d = self.__normalize_keys(ids=top1_indices_qb)  # Recall that b=1 here

            # Updating responses (the response with the updated key is recomputed, for each neuron)
            response_winning_bq1 = torch.matmul(x_bd, normalized_winning_keys_q1d.view(q, d).t()).view(b, q, 1)
            top_responses_bqk[:, :, 0] = response_winning_bq1.squeeze()

        elif b > 1:

            # Normalizing all the keys (for simplicity - with large batch sizes it is likely easier/faster)
            self.__normalize_keys()

            # Recomputing all responses, re-determining the top-responses
            top_responses_bqk, top_indices_bqk = self.__top_k_attention(x_bd)

            # Re-transposing top-1 indices
            top1_indices_qb = top_indices_bqk[..., 0].t()

        return top_responses_bqk, top_indices_bqk, top1_indices_qb

    def __str__(self):
        s = "- q = " + str(self.q) + "\n"
        s += "- d = " + str(self.d) + "\n"
        s += "- m = " + str(self.m) + "\n"
        s += "- u = " + str(self.u) + "\n"
        s += "- delta = " + str(self.delta) + "\n"
        s += "- gamma_alpha = " + str(self.gamma_alpha) + "\n"
        s += "- tau_alpha = " + str(self.tau_alpha) + "\n"
        s += "- tau_mu = " + str(self.tau_mu) + "\n"
        s += "- tau_eta = " + str(self.tau_eta) + "\n"
        s += "- upd_m = " + str(self.upd_m) + "\n"
        s += "- upd_k = " + str(self.upd_k) + "\n"
        s += "- beta_k = " + str(self.beta_k) + "\n"
        s += "- psi_fn = " + self.psi_fn + "\n"
        s += "- scramble = " + str(self.scramble)
        return s

    def __debug_pre_scrambling(self, top_indices_bqk, top_alphas_bqk, scramble_q,
                               weak_keys_mask_qm, weak_batch_elements_q, x_bd):
        b, q, k, d, m = top_indices_bqk.shape[0], self.q, self.delta, self.d, self.m
        mu_qm, eta_qm = self.mu, self.eta

        # Finding the indices of the candidate keys to scramble (one per neuron)
        scramble_candidates_keys_q = torch.max(eta_qm * weak_keys_mask_qm.to(torch.float), dim=1)[1]

        print("*** __debug_pre_scrambling ***")
        print("No scrambling is applied at all...not yet...")
        s = ""
        torch.set_printoptions(profile='full', linewidth=2000)
        for i in range(0, b):
            if i > 0:
                s += "\n"
            for j in range(0, q):
                if j > 0:
                    s += "\n"
                s += "[batch element " + str(i)
                s += ", neuron " + str(j) + "] " if q > 1 else "]"
                s += "\n- tau_alpha, tau_mu, tau_eta: " + str(self.tau_alpha) + ", "  \
                     + str(self.tau_mu) + ", " + str(self.tau_eta)
                s += "\n- mu:       " + str(mu_qm[j, :])
                s += "\n- eta:      " + str(eta_qm[j, :])
                s += "\n- top-keys (alphas): "
                for a in range(0, k):
                    s += str(top_indices_bqk[i, j, a].item())
                    s += " ({0:.3g})".format(top_alphas_bqk[i, j, a].item())
                    if a < k - 1:
                        s += ", "
                s += "\n- scramble? " + str(scramble_q[j].item())
                if scramble_q[j].item() is True:
                    s += "\n  - what key? " + str(scramble_candidates_keys_q[j].item())
                    s += "\n  - with what batch element? " + str(weak_batch_elements_q[j].item())
                    s += "\n  - data of such element? "
                    s += str(x_bd[weak_batch_elements_q[j], 0:min(3, d)])
                    if d > 3:
                        s += " ~printing only the first 3 components"
        print(s)
        torch.set_printoptions(profile='default')

    def __debug_changes_with_respect_to(self, K_initial_qmd, M_initial_qmd, mu_initial_qm, eta_initial_qm,
                                        top1_indices_initial_qb, top1_indices_qb, msg=None):
        q, m, b, u, d = self.q, self.m, top1_indices_initial_qb.shape[1], self.u, self.d

        changed_keys_qm = torch.greater(torch.max(torch.abs(self.K - K_initial_qmd), dim=2)[0], 1e-5)
        changed_memories_qm = torch.greater(torch.max(torch.abs(self.M - M_initial_qmd), dim=2)[0], 0.)
        changed_top1_indices_qb = torch.greater(torch.abs(top1_indices_initial_qb.to(torch.float) -
                                                          top1_indices_qb.to(torch.float)), 0.)
        changed_mus_qm = torch.greater(torch.abs(self.mu - mu_initial_qm), 0.)
        changed_etas_qm = torch.greater(torch.abs(self.eta - eta_initial_qm), 0.)

        print("*** __debug_changes_with_respect_to ***")
        if msg is not None:
            print(msg)
        s = ""
        for j in range(0, q):
            if j > 0:
                s += "\n"
            s += "[neuron " + str(j) + "]"
            num_changed_keys = torch.sum(changed_keys_qm[j, :]).item()
            num_changed_memories = torch.sum(changed_memories_qm[j, :]).item()
            num_changed_top1_indices = torch.sum(changed_top1_indices_qb[j, :]).item()
            num_changed_mus = torch.sum(changed_mus_qm[j, :]).item()
            num_changed_etas = torch.sum(changed_etas_qm[j, :]).item()
            s += "\n- #changed keys: " + str(num_changed_keys)
            if num_changed_keys > 0:
                if d > 3:
                    s += " ~printing only the first 3 components"
                for t in range(0, m):
                    if changed_keys_qm[j, t].item() is True:
                        s += "\n  - key " + str(t) + ": " + str(self.K[j, t, 0:min(3, d)])
                        s += " (it was: " + str(K_initial_qmd[j, t, 0:min(3, d)]) + ")"
            s += "\n- #changed memory units: " + str(num_changed_memories)
            if num_changed_memories > 0:
                if u > 3:
                    s += " ~printing only the first 3 components"
                for t in range(0, m):
                    if changed_memories_qm[j, t].item() is True:
                        s += "\n  - mem " + str(t) + ": " + str(self.M[j, t, 0:min(3, u)])
                        s += " (it was: " + str(M_initial_qmd[j, t, 0:min(3, u)]) + ")"
            s += "\n- #changed top1 indices: " + str(num_changed_top1_indices)
            if num_changed_top1_indices > 0:
                for i in range(0, b):
                    if changed_top1_indices_qb[j, i].item() is True:
                        s += "\n  - batch element " + str(i) + ": " + str(top1_indices_qb[j, i].item())
                        s += " (it was: " + str(top1_indices_initial_qb[j, i].item()) + ")"
            s += "\n- #changed mus: " + str(num_changed_mus)
            if num_changed_mus > 0:
                for t in range(0, m):
                    if changed_mus_qm[j, t].item() is True:
                        s += "\n  - mu " + str(t) + ": " + str(self.mu[j, t].item())
                        s += " (it was: " + str(mu_initial_qm[j, t].item()) + ")"
            s += "\n- #changed etas: " + str(num_changed_etas)
            if num_changed_etas > 0:
                for t in range(0, m):
                    if changed_etas_qm[j, t].item() is True:
                        s += "\n  - eta " + str(t) + ": " + str(self.eta[j, t].item())
                        s += " (it was: " + str(eta_initial_qm[j, t].item()) + ")"
        print(s)
