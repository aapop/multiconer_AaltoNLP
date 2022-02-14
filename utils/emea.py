import torch


def ensemble(scores, weights):
    output = weights[0]*scores[0]
    for i in range(1, len(weights)):
        output += weights[i]*scores[i]
    return output

def emea_ensemble(scores, steps=10, lr=10):
    adapter_weights = [torch.FloatTensor([1 for _ in range(len(scores))])]
    for _ in range(steps):
        for w in adapter_weights: w.requires_grad=True
        normed_adapter_weights = [torch.nn.functional.softmax(w, dim=0) for w in adapter_weights][0]
        kept_logits = ensemble(scores, normed_adapter_weights)
        entropy = torch.nn.functional.softmax(kept_logits, dim=1)*torch.nn.functional.log_softmax(kept_logits, dim=1)
        entropy = -entropy.sum() / kept_logits.size(0)
        grads = torch.autograd.grad(entropy, adapter_weights)
        for i, w in enumerate(adapter_weights):
            adapter_weights[i] = adapter_weights[i].data - lr*grads[i].data
            
    normed_adapter_weights = [torch.nn.functional.softmax(w, dim=0) for w in adapter_weights][0]
    return ensemble(scores, normed_adapter_weights)