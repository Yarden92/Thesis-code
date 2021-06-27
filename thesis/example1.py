import numpy as np

# import thesis.src.visualizer
from FNFTpy.FNFTpy import kdvv

print("\n\nkdvv example")

# set values
D = 256
tvec = np.linspace(-1, 1, D)
q = np.zeros(D, dtype=np.complex128)
q[:] = 2.0 + 0.0j
Xi1 = -2
Xi2 = 2
M = 1024
Xivec = np.linspace(Xi1, Xi2, M)
rq = np.real(q)
iq = np.imag(q)
# thesis.src.visualizer.my_plot(tvec, rq, tvec, iq, title="q[t]", output_name="before.jpg", xlabel="t[sec]",
#                               legend=['real', 'imag'])

# call function
res = kdvv(q, tvec, M, Xi1=Xi1, Xi2=Xi2)

# print results
print("\n---- options used ----")
print(res['options'])
print("\n----- results -----")
print(f"FNFT return value: {res['return_value']} (should be 0)")
print("continuous spectrum: ")
for i in range(len(res['cont'])):
    a = np.round(Xivec[i], 4)
    b = np.round(np.real(res['cont'][i]), 6)
    c = np.round(np.imag(res['cont'][i]), 6)
    print(f"{i} : Xi={a} {b}j")

rQ = np.real(res['cont'])
iQ = np.imag(res['cont'])
# thesis.src.visualizer.my_plot(Xivec, rQ, Xivec, iQ, title="Q[xi]", output_name="after.jpg", xlabel="xi",
#                               legend=['real', 'imag'])
# thesis.src.visualizer.my_plot(Xivec, np.abs(res['cont']), title="|Q[xi]|", output_name="abs.jpg", xlabel="xi")
