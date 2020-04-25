#!/usr/bin/python3.8
import numpy
import io
# slightly modified version of ikrlib
from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm

t_train_sound__dict = wav16khz2mfcc("target_train")
t_train_sound = numpy.vstack(list(t_train_sound__dict.values()))
nt_train_sound__dict = wav16khz2mfcc("non_target_train")
nt_train_sound = numpy.vstack(list(wav16khz2mfcc("non_target_train").values()))
eval_sound__dict = wav16khz2mfcc("eval")
eval_sound = list(eval_sound__dict.values())
eval_sound_names = [x.split("\\")[1].split(".")[0]
                    for x in list(eval_sound__dict.keys())]

P_t = 0.5
M_t = 64
MUs_t = t_train_sound[numpy.random.randint(1, len(t_train_sound), M_t)]
COVs_t = [numpy.var(t_train_sound, axis=0)] * M_t
Ws_t = numpy.ones(M_t) / M_t

P_nt = 1 - P_t
M_nt = M_t
MUs_nt = nt_train_sound[numpy.random.randint(1, len(nt_train_sound), M_nt)]
COVs_nt = [numpy.var(nt_train_sound, axis=0)] * M_nt
Ws_nt = numpy.ones(M_nt) / M_nt

n = 32
for i in range(n):
    [Ws_t, MUs_t, COVs_t, TTL_t] = train_gmm(
        t_train_sound, Ws_t, MUs_t, COVs_t)
    [Ws_nt, MUs_nt, COVs_nt, TTL_nt] = train_gmm(
        nt_train_sound, Ws_nt, MUs_nt, COVs_nt)

    print("Training iteration: " + str(i + 1) + "/" + str(n))

with open("GMM_speech_results", "w") as f:
    for i, eval in enumerate(eval_sound):
        ll_t = logpdf_gmm(eval, Ws_t, MUs_t, COVs_t)
        ll_nt = logpdf_gmm(eval, Ws_nt, MUs_nt, COVs_nt)
        val = (sum(ll_t) + numpy.log(P_t)) - (sum(ll_nt) + numpy.log(P_nt))

        f.write(eval_sound_names[i] + " " + str(val) +
                " " + ("1" if val > 0 else "0") + "\n")
