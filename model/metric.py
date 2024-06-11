import torch

def get_prediction(output, threshold):
    with torch.no_grad():
        if threshold is None:
            pred = torch.argmax(output, dim=1)
        elif len(output.shape) > 1:
            pred = (output > threshold).long()
            pred = torch.any(pred,dim=1).long()
        else:
            pred = (output > threshold).long()
            pred = torch.squeeze(pred)
    return pred

def accuracy(output, target, threshold=0.5):
    with torch.no_grad():
        pred = get_prediction(output, threshold)
        assert pred.shape[0] == target.shape[0]

    return torch.sum(pred == target) / target.shape[0]


