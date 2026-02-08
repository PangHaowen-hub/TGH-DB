import torch
import torch.nn as nn

class DiffusionBridge(nn.Module):
    def __init__(
            self,
            n_steps,
            gamma,
            n_recursions,
            consistency_threshold,
            beta_start=None, 
            beta_end=None,
        ):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma 
        self.n_recursions = n_recursions
        self.consistency_threshold = consistency_threshold

        self.register_buffer('alphas_bar', torch.linspace(1.0, 0.0, n_steps + 1))
        
        
    def q_sample(self, t, x0, mu_atlas, sigma_atlas):
        shape = [-1] + [1] * (x0.ndim - 1)
        a_bar = self.alphas_bar[t].view(shape)
        
        sqrt_a_bar = torch.sqrt(a_bar)
        mu_t = sqrt_a_bar * x0 + (1 - sqrt_a_bar) * mu_atlas
        
        epsilon = torch.randn_like(x0)
        std_t = torch.sqrt(1 - a_bar) * sigma_atlas * self.gamma
        
        return (mu_t + std_t * epsilon).detach()


    @torch.no_grad()
    def ddim_predict_step(self, y_t1, y_fl, mu_atlas, sigma_atlas, model, skip_steps=20):
        times = list(range(self.n_steps, 0, -skip_steps))
        if times[-1] != 1: times.append(1)
        
        device = y_t1.device
        t_T = torch.full((y_t1.shape[0],), self.n_steps, device=device, dtype=torch.long)
        x_t = self.q_sample(t_T, torch.zeros_like(mu_atlas), mu_atlas, sigma_atlas)

        for i in range(len(times)):
            t_val = times[i]
            prev_t_val = times[i+1] if i+1 < len(times) else 0
            
            t = torch.full((y_t1.shape[0],), t_val, device=device, dtype=torch.long)
            
            img_input_t1 = torch.cat([x_t, y_t1], dim=1)
            img_input_fl = torch.cat([x_t, y_fl], dim=1)
            x0_pred = model(img_input_t1, img_input_fl, t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            shape = [-1] + [1] * (x0_pred.ndim - 1)
            a_bar_t = self.alphas_bar[t_val].view(shape)
            eps_t = ((x_t - mu_atlas) - torch.sqrt(a_bar_t) * (x0_pred - mu_atlas)) / (torch.sqrt(1 - a_bar_t) * sigma_atlas * self.gamma + 1e-10)

            a_bar_prev = self.alphas_bar[prev_t_val].view(shape)
            
            pred_dir_xt = torch.sqrt(1 - a_bar_prev) * sigma_atlas * self.gamma * eps_t
            
            x_t = torch.sqrt(a_bar_prev) * (x0_pred - mu_atlas) + mu_atlas + pred_dir_xt

        return x_t