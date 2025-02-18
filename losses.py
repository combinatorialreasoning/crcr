import torch
import gin 
import torch.nn as nn

@gin.configurable
def mrn_loss(repr_0, repr_T, **unused_kwargs):
    x, psi_0 = repr_0
    _, y = repr_T
    
    eps = 1e-8
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = torch.max(torch.nn.functional.relu(x_prefix - y_prefix), axis=-1).values
    l2_component = torch.sqrt(torch.square(x - y_suffix).sum(axis=-1) + eps)
    assert max_component.shape == l2_component.shape
    
    
    loss = max_component + l2_component
    
    pdist = torch.sum((x_suffix[:, None] - y_suffix[None])**2, axis=-1)
    accuracy = torch.sum(torch.argmin(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
    
    l2 = (torch.mean(x_suffix**2) + torch.mean(y_suffix**2)) / 2
    
    metrics = {
                    "loss": loss.mean(), 
                    "max_component": max_component.mean(),
                    "l2_component": l2_component.mean(),
                    
                    "l2": l2,
                    'accuracy': accuracy
        }
    
    return loss.mean(), metrics

@gin.configurable
def get_mrn_dist(x, y, coeff=1):
    d = x.shape[-1]
    eps = 1e-8
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = torch.max(torch.nn.functional.relu(x_prefix - y_prefix), axis=-1).values
    # max_component = torch.sqrt(torch.square(x_prefix - y_prefix).sum(axis=-1) + eps)
    l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    
    return coeff*max_component + l2_component

@gin.configurable
def vanilla_loss(repr_0, repr_T, normalize=False, tau=None, unif_const=1, distance_fun='l2', exclude_diagonal = False, log_lambda=None, c=None, eps=10e-8, loss_type='symmetric', logsumexp_penalty=0, W=None, **unused_kwargs):
        phi_0, psi_0 = repr_0
        _, psi_T = repr_T

        if tau is None:
            tau = 1 / (phi_0.shape[1] ** 0.5)

        if normalize:
            phi_0 = nn.functional.normalize(phi_0, p=2, dim=1)
            psi_0 = nn.functional.normalize(psi_0, p=2, dim=1)
            psi_T = nn.functional.normalize(psi_T, p=2, dim=1)
                

        l2 = (torch.mean(psi_T**2) + torch.mean(psi_0**2)) / 2
        I = torch.eye(psi_0.shape[0], device=psi_0.device)

        if distance_fun == 'l2':
            l_align = torch.sum((psi_T - phi_0)**2, axis=1) * tau
            pdist = torch.sum((psi_T[:, None] - phi_0[None])**2, axis=-1) * tau / psi_T.shape[-1]

            accuracy = torch.sum(torch.argmin(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
            if exclude_diagonal:
                 pdist = pdist * (1 - I)

        # TODO fix dot product
        elif distance_fun == 'dot':    
            l_align = -torch.matmul(psi_T, phi_0.T)[torch.eye(psi_T.shape[0]).to(bool)] * tau
            pdist = -torch.sum(psi_T[:, None] * phi_0[None], axis=-1) * tau / psi_T.shape[-1]
            accuracy = torch.sum(torch.argmax(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
            if exclude_diagonal:
                 pdist = pdist * (1 - I)
        
        elif distance_fun == 'l1':
            l_align = torch.sum(torch.abs(psi_T - phi_0), axis=1) * tau
            pdist = torch.sum((psi_T[:, None] - phi_0[None])**2, axis=-1) * tau / psi_T.shape[-1]
            
            accuracy = torch.sum(torch.argmin(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
            if exclude_diagonal:
                 pdist = pdist * (1 - I)
                 
        elif distance_fun == 'l22':
            l_align = ((torch.sum((psi_T - phi_0)**2, axis=1) + eps)**0.5) * tau
            pdist = ((torch.sum((psi_T[:, None] - phi_0[None])**2, axis=-1) + eps)**0.5) * tau / psi_T.shape[-1]
            accuracy = torch.sum(torch.argmin(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
            if exclude_diagonal:
                 pdist = pdist * (1 - I)
                 
        elif distance_fun == 'mrn':
            l_align = (get_mrn_dist(psi_T, phi_0)) * tau
            pdist = (get_mrn_dist(psi_T[:, None], phi_0[None])) * tau / psi_T.shape[-1]
            
            accuracy = torch.sum(torch.argmin(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
            if exclude_diagonal:
                 pdist = pdist * (1 - I)   

        elif distance_fun == 'learned':
            psi_T = torch.matmul(psi_T, W)

            l_align = torch.matmul(psi_T, phi_0.T)[torch.eye(psi_T.shape[0]).to(bool)] * tau
            pdist = torch.sum(psi_T[:, None] * phi_0[None], axis=-1) * tau / psi_T.shape[-1]
            
            accuracy = torch.sum(torch.argmax(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
            if exclude_diagonal:
                 pdist = pdist * (1 - I)      
        
        else: 
             raise ValueError()

        if loss_type == 'symmetric':
            l_unif = (torch.logsumexp(-pdist, axis=1) + torch.logsumexp(-pdist.T, axis=1)) / 2.0
        elif loss_type == 'forward':
            l_unif = torch.logsumexp(-pdist, axis=1) 
        elif loss_type == 'backward':
            l_unif = torch.logsumexp(-pdist.T, axis=1) 
        else:
            raise ValueError()
            

        

        loss = l_align + psi_T.shape[-1] * l_unif

        if logsumexp_penalty > 0:
            logits_ = pdist
            big = 100
            I = torch.eye(logits_.shape[0], device=logits_.device)  # Ensure the identity matrix is on the same device

            eps = 1e-6
            logsumexp = torch.logsumexp(logits_ + eps, dim=1)
            loss += logsumexp_penalty * torch.mean(logsumexp**2)


        if log_lambda is not None:
            dual_loss = log_lambda * (c - l2.detach())

            loss = loss.mean() + torch.exp(log_lambda).detach() * l2 + dual_loss
            metrics = {
                    "l_unif": l_unif.mean(), 
                    "l_align": l_align.mean(),
                    "lambda": log_lambda.item(),
                    "dual_loss": dual_loss,
                    
                    "accuracy": accuracy,
                    "l2": l2,
                    "loss": loss
        }
        else:
            loss = loss.mean()

            metrics = {
                        "l_unif": l_unif.mean(), 
                        "l_align": l_align.mean(),

                        "accuracy": accuracy,
                        "l2": l2,
                        "loss": loss
            }
        del psi_0
        del psi_T

        return loss, metrics

@gin.configurable
def vanilla_loss_effemb(repr_0, repr_T, normalize=False, tau=1, **unused_kwargs):
        phi_0, psi_0 = repr_0
        _, psi_T = repr_T

        phi_0 = phi_0.cpu()
        psi_0 = psi_0.cpu()
        psi_T = psi_T.cpu()

        if normalize:
            phi_0 = nn.functional.normalize(phi_0, p=2, dim=1)
            psi_0 = nn.functional.normalize(psi_0, p=2, dim=1)
            psi_T = nn.functional.normalize(psi_T, p=2, dim=1)
                

        l2 = (torch.mean(psi_T**2) + torch.mean(psi_0**2)) / 2

        logits = torch.matmul(phi_0, psi_T.T) * tau

        print("LOGITS", logits)

        accuracy = torch.sum(torch.argmax(logits, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
        print('accuracy', accuracy)
        batch_size = psi_0.size(0)

        labels = torch.arange(batch_size)
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels) + loss_fn(logits.T, labels) / 2
        loss = loss.mean()


        metrics = {
                    # "l_unif": l_unif.mean(), 
                    # "l_align": l_align.mean(),
                    "accuracy": accuracy,
                    "l2": l2,
                    "loss": loss
        }
        del psi_0
        del psi_T

        return loss, metrics

@gin.configurable
def log_lambda_loss(repr_0, repr_T, log_lambda, c):
        phi_0, psi_0 = repr_0
        _, psi_T = repr_T

        phi_0 = phi_0.cpu()
        psi_0 = psi_0.cpu()
        psi_T = psi_T.cpu()

        l2 = (torch.mean(psi_T**2) + torch.mean(psi_0**2)) / 2
        I = torch.eye(psi_0.shape[0], device=psi_0.device)
        l_align = torch.sum((psi_T - phi_0)**2, axis=1) * 2

        pdist = torch.mean((psi_T[:, None] - phi_0[None])**2, axis=-1)
        l_unif = (torch.logsumexp(-(pdist * (1 - I)), axis=1) + torch.logsumexp(-(pdist.T * (1 - I)), axis=1)) / 2.0

        loss = l_align + l_unif

        accuracy = torch.sum(torch.argmin(pdist, axis=1) == torch.arange(psi_0.shape[0], device=psi_0.device)) / psi_0.shape[0]
        dual_loss = log_lambda * (c - l2.detach())

        full_loss = loss.mean() + torch.exp(log_lambda).detach() * l2 + dual_loss
        metrics = {
                    "l_unif": l_unif.mean(),
                    "l_align": l_align.mean(),
                    "accuray": accuracy,
                    "l2": l2,
                    "log_lambda": log_lambda.item(),
                    "dual_loss": dual_loss.mean(),
                    "full_loss": full_loss
        }
        del psi_0
        del psi_T

        return full_loss, metrics

def main():
    a = torch.rand((1, 54))
    b = torch.rand((1, 54))
    x1 = (a, a)
    x2 = (b, b)

    print(vanilla_loss(x1, x2))
    print(vanilla_loss_effemb(x1, x2))
if __name__ == "__main__":
    main()