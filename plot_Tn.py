import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Tn_all = pd.read_csv('exact_data.dat', dtype = float, sep = ',')
Tn_all = Tn_all.to_numpy()

plt.figure()
plt.subplot(311)
#plt.title('Tn-marg')
plt.plot(Tn_all[:,0], Tn_all[:,1], '-bo', label = "Tn-chi")
plt.ylabel("Kin (a.u)")
plt.legend()

plt.subplot(312)
#plt.title('Tn-$\Phy$')
plt.plot(Tn_all[:,0], Tn_all[:,2], '-ko', label = "Tn-Phy")
plt.ylabel("Kin (a.u)")
plt.legend()

plt.subplot(313)
plt.plot(Tn_all[:,0], Tn_all[:,3], '-ro', label = "Tn-Psi")
plt.ylabel("Kin (a.u)")
plt.xlabel("Time (fs)")

plt.legend()
plt.show()