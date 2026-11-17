"""Ordinary independent-frame single-template spectral explanation."""
import numpy as np


def analyze(templates, spectrum, ridge=0.5):
    dictionary=np.asarray(templates,dtype=float)
    y=np.asarray(spectrum,dtype=float)
    activations=np.zeros((dictionary.shape[1],y.shape[1]))
    for frame in range(y.shape[1]):
        amplitudes=np.maximum(dictionary.T@y[:,frame],0)/(np.square(dictionary).sum(axis=0)+ridge)
        costs=[]
        for pitch in range(dictionary.shape[1]):
            error=dictionary[:,pitch]*amplitudes[pitch]-y[:,frame]
            costs.append(.5*float(error@error)+ridge/2*amplitudes[pitch]**2)
        best=int(np.argmin(costs))
        activations[best,frame]=amplitudes[best]
    residual=dictionary@activations-y
    return {'activations':activations.tolist(),
            'objective':float(.5*np.square(residual).sum()+ridge/2*np.square(activations).sum())}
