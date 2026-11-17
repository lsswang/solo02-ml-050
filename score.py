"""Ordinary threshold/merge transcription, always using the first voice."""


def transcribe(activations,onsets,request):
    notes=[]
    for pitch,row in enumerate(activations):
        start=None
        for t,value in enumerate(list(row)+[0.]):
            if value>=.5 and start is None:
                start=t
            elif value<.5 and start is not None:
                notes.append({'voice':0,'pitch':request['pitches'][pitch],
                              'start':start,'end':t})
                start=None
    return {'notes':notes}
