#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estructura y evolución estelar: Ecuación de Lane-Emden y modelos politrópicos
solares.

@author: Alejandro Cerón Fernández
"""

from scipy . integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib import colormaps
from astropy.constants import M_sun, R_sun, G, R, h, m_e, m_p, c, sigma_sb, k_B
from astropy.io.ascii import read
from astropy import units as u
from astropy.table import QTable
from scipy.interpolate import interp1d
import matplotlib.lines as mlines
import os
import shutil

############################ LaTeX rendering ##############################
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX rendering
plt.rc('font', size=16)  # Adjust size to your preference
# Define the LaTeX preamble with siunitx
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \sisetup{
              detect-family,
              separate-uncertainty=true,
              output-decimal-marker={.},
              exponent-product=\cdot,
              inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            \DeclareSIUnit{\year}{yr}
            \DeclareSIUnit{\dyn}{dyn}
            \DeclareSIUnit{\mag}{mag}
            \usepackage{sansmath}  % Allows sans-serif in math mode

            '''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
})
# \sansmath para usar fuente sans en modo matemático

#############################################################################
# Carpeta donde están los modelos de evolución:
data_dir = "Selection_Geneva_Models"
# Carpeta donde guardaremos los plots:
plots_dir = "products"
# Creamos la carpeta con los plots:
if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)    
os.makedirs(plots_dir)
#############################################################################
# Creamos una tabla donde guardar características de las estrellas:
star_params = QTable()
star_params["Mass"] = np.flip([0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 7, 9, 15, 25, 40]*u.solMass)

############################################################################
def read_table(model_name:str):
    # Leemos la tabla y obtenemos una Qtable para tener unidades en las columnas:
    table = read (os.path.join(data_dir, model_name) , data_start = 2) 
    table = QTable(table)
    # Adjudicamos unidades:
    return table

# Lista de archivos en el directorio:
files = os.listdir(data_dir)
# Diccionario con los datos:
data = {os.path.basename(os.path.splitext(f)[0]): read_table(f) for f in files}
# Ordenamos el dicionario poniendo las estrellas más frías primero:
data = dict(
    sorted(data.items(), key=lambda item: item[1]["lg(Teff)"][0], reverse=True)
)

def compute_TAMS(table, umbral):
    """
    Calcula la línea en que la estrella llega al TAMS
    """
    condition = table["1H_cen"] < umbral
    line = table["line"][condition][0]

    return line-1

def compute_BL(table, radii):
    """
    Calcula la línea en que la estrella llega al Blue Loop encontrando el instante en
    que el radio es máximo
    """
    condition = (radii == np.max(radii))
    line = table["line"][condition][0]

    return line-1

def compute_AGB(table, umbral):
    """
    Calcula la línea en que la estrella llega a la fase AGB
    """
    condition = table["4He_cen"] < umbral
    line = table["line"][condition][0]

    return line-1



# 1 #######################################################################################
"""
Plot the time evolution of the central hydrogen and helium abundances for stars
with M = 5 M⊙ and M = 9 M⊙ . How long does the main sequence last for the
two stars? Is this the result that you expected? Why?
"""
###########################################################################################

fig1, ax1 = plt.subplots(figsize=(17, 10))

# Abundancias de H
ax1.plot(data["M005Z14V0"]["time"]/1e6, data["M005Z14V0"]["1H_cen"], color="red", label=r"$M=5 \mathrm{M}_{\odot}$, H")
ax1.plot(data["M009Z14V0"]["time"]/1e6, data["M009Z14V0"]["1H_cen"], color="blue", label=r"$M=9 \mathrm{M}_{\odot}$, H ")
# Abundancias de He
ax1.plot(data["M005Z14V0"]["time"]/1e6, data["M005Z14V0"]["4He_cen"], color="red", linestyle="--", label=r"$M=5 \mathrm{M}_{\odot}$, He")
ax1.plot(data["M009Z14V0"]["time"]/1e6, data["M009Z14V0"]["4He_cen"], color="blue", linestyle="--", label=r"$M=9 \mathrm{M}_{\odot}$, He ")

ax1.axvline(data["M005Z14V0"]["time"][compute_TAMS(data["M005Z14V0"], 1e-4)]/1e6, color="black",linestyle="dashdot", linewidth=2, 
label=r"TAMS", zorder=0, alpha=0.9)
ax1.axvline(data["M009Z14V0"]["time"][compute_TAMS(data["M009Z14V0"], 1e-4)]/1e6, color="black",linestyle="dashdot", linewidth=2,
zorder=0.1, alpha=0.9)

# Ajustes del plot:
ax1.set_xlabel(r"$t$ [$\unit{\mega\year}$]", fontsize=30)
ax1.set_ylabel(r"Central mass fraction", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlim(0 , None)
ax1.set_ylim(0, 1)

ax1.legend(loc=(0.28,0.63), fontsize=24)

fig1.savefig(os.path.join(plots_dir, "Apartado_1.pdf"), format="pdf", bbox_inches='tight')


# 2 #######################################################################################
"""
For the models with M = 0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 7, 9, 15, 25, 40 M⊙ :
"""
###########################################################################################

### a)  Plot the evolutionary tracks in an HR diagram.
fig2, ax2 = plt.subplots(figsize=(12, 17))
cmap = colormaps.get_cmap("jet")
colors = cmap(np.linspace(0, 1, len(data)))
for i, star in enumerate(data):
    ax2.plot(data[star]["lg(Teff)"], data[star]["lg(L)"], color=colors[i])
    ax2.text(data[star]["lg(Teff)"][0]+0.22, data[star]["lg(L)"][0], rf"${star_params["Mass"][i].value}$ $\mathrm{{M}}_{{\odot}}$", color=colors[i], fontsize=24)

# Ajustes del plot:
ax2.set_xlim(None, 4.9)
ax2.invert_xaxis()
ax2.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}} [\unit{\kelvin}])$", fontsize=30)
ax2.set_ylabel(r"$\log_{10}(L/\mathrm{L}_{\odot})$", fontsize=30)
ax2.tick_params(axis='both', which='major', labelsize=24)

### b) Plot the ZAMS and the TAMS. What is the cause of the different main
### sequence behaviours for stars in different mass ranges?
for i, star in enumerate(data):
    # ZAMS:
    ax2.plot(data[star]["lg(Teff)"][0], data[star]["lg(L)"][0], marker=">", color="black", 
             markersize=10, linewidth=0, label="ZAMS" if i==0 else None)
    # TAMS:
    ax2.plot(data[star]["lg(Teff)"][compute_TAMS(data[star], 1e-4)], data[star]["lg(L)"][compute_TAMS(data[star], 1e-4)], 
             marker="<", color="black", markersize=10, linewidth=0, label="TAMS" if i==0 else None )
ax2.legend(loc="lower left", fontsize=24)

fig2.savefig(os.path.join(plots_dir, "Apartado_2_ab.pdf"), format="pdf", bbox_inches='tight')

### c) What is the age of the stars when they reach the ZAMS? 
### d) What is the age of the stars as they leave the main sequence?
star_params["ZAMS_age"] = [table["time"][0] for (table_name,table) in data.items()]*u.year
star_params["TAMS_age"] = [table["time"][compute_TAMS(data[table_name], 1e-4)] for (table_name,table) in data.items()]*u.year
# Pretty print the table
print("=" * 40)
print(f"{'Mass':<10}{'ZAMS_age':<15}{'TAMS_age':<15}")
print("=" * 40)
for row in star_params:
    print(f"{row['Mass'].value:<10.2f}{row['ZAMS_age'].value:<15.4e}{row['TAMS_age'].value:<15.4e}")
print("=" * 40)

### e) Plot the L − M relation for ZAMS stars. Compare the slopes to those
### expected from the homology relations.
fig3, ax3 = plt.subplots(figsize=(11, 11))

log_mass_ZAMS = [np.log10(data[star]["mass"][0]) for star in data] # masas solares
log_L_ZAMS = [data[star]["lg(L)"][0] for star in data] # luminosidades solares
ax3.plot(log_mass_ZAMS, log_L_ZAMS, linewidth=0, color="black", markersize=15, marker="s")

# Fit a los 3 primeros puntos:
pol_1 = np.polyfit(log_mass_ZAMS[:3], log_L_ZAMS[:3], 1)
mm_1 = np.linspace(log_mass_ZAMS[0], log_mass_ZAMS[2], 200)
ll_1 = np.polyval(pol_1, mm_1)
ax3.plot(mm_1,ll_1, linewidth=2, color="blue")
ax3.text(log_mass_ZAMS[2], log_L_ZAMS[2]+1, rf"$s_3={pol_1[0]:.2f}$", fontsize=24, color="blue")

# Fit a los 4 siguientes puntos:
pol_2 = np.polyfit(log_mass_ZAMS[3:7], log_L_ZAMS[3:7], 1)
mm_2 = np.linspace(log_mass_ZAMS[3], log_mass_ZAMS[6], 200)
ll_2 = np.polyval(pol_2, mm_2)
ax3.plot(mm_2,ll_2, linewidth=2, color="green")
ax3.text(log_mass_ZAMS[6], log_L_ZAMS[6]+1.2, rf"$s_2={pol_2[0]:.2f}$", fontsize=24, color="green")

# Fit a los 5 últimos puntos:
pol_2 = np.polyfit(log_mass_ZAMS[7:], log_L_ZAMS[7:], 1)
mm_2 = np.linspace(log_mass_ZAMS[7], log_mass_ZAMS[-1], 200)
ll_2 = np.polyval(pol_2, mm_2)
ax3.plot(mm_2,ll_2, linewidth=2, color="red")
ax3.text(log_mass_ZAMS[10], log_L_ZAMS[10]+1.5, rf"$s_1={pol_2[0]:.2f}$", fontsize=24, color="red")

ax3.set_xlabel(r"$\log_{10}(M/\mathrm{M}_{\odot})$", fontsize=30)
ax3.set_ylabel(r"$\log_{10}(L/\mathrm{L}_{\odot})$", fontsize=30)
ax3.tick_params(axis='both', which='major', labelsize=24)
fig3.savefig(os.path.join(plots_dir, "Apartado_2_e.pdf"), format="pdf", bbox_inches='tight')

### f)  Plot the ρc − M relation for ZAMS stars. Compare the slopes to those
### expected from the homology relations.
fig4, ax4 = plt.subplots(figsize=(11, 11))

log_rhoc = [data[star]["lg(rhoc)"][0] for star in data] # g/cm^3
ax4.plot(log_mass_ZAMS, log_rhoc, linewidth=0, color="black", markersize=15, marker="s")

# Fit a los 3 primeros puntos:
pol_1 = np.polyfit(log_mass_ZAMS[:3], log_rhoc[:3], 1)
mm_1 = np.linspace(log_mass_ZAMS[0], log_mass_ZAMS[2], 200)
pp_1 = np.polyval(pol_1, mm_1)
ax4.plot(mm_1,pp_1, linewidth=2, color="blue")
ax4.text(log_mass_ZAMS[2]+0.15, log_rhoc[2]+0.03, rf"$s_3={pol_1[0]:.2f}$", fontsize=24, color="blue")

# Fit a los 6 siguientes puntos:
pol_2 = np.polyfit(log_mass_ZAMS[3:9], log_rhoc[3:9], 1)
mm_2 = np.linspace(log_mass_ZAMS[3], log_mass_ZAMS[8], 200)
pp_2 = np.polyval(pol_2, mm_2)
ax4.plot(mm_2,pp_2, linewidth=2, color="green")
ax4.text(log_mass_ZAMS[5]+0.15, log_rhoc[5]+0.03, rf"$s_2={pol_2[0]:.2f}$", fontsize=24, color="green")

# Fit a los 4 últimos puntos:
pol_3 = np.polyfit(log_mass_ZAMS[9:], log_rhoc[9:], 1)
mm_3 = np.linspace(log_mass_ZAMS[9], log_mass_ZAMS[-1], 200)
pp_3 = np.polyval(pol_3, mm_3)
ax4.plot(mm_3,pp_3, linewidth=2, color="red")
ax4.text(log_mass_ZAMS[10]-0.16, log_rhoc[10]-0.16, rf"$s_1={pol_3[0]:.2f}$", fontsize=24, color="red")

ax4.set_xlabel(r"$\log_{10}(M/\mathrm{M}_{\odot})$", fontsize=30)
ax4.set_ylabel(r"$\log_{10}(\rho_{{\mathrm{c}}} \, [\unit{\gram\per\centi\meter\cubed}])$", fontsize=30)
ax4.tick_params(axis='both', which='major', labelsize=24)
fig4.savefig(os.path.join(plots_dir, "Apartado_2_f.pdf"), format="pdf", bbox_inches='tight')

### g)  Plot the ρc − M relation for ZAMS stars. Compare the slopes to those
### expected from the homology relations.
fig5, ax5 = plt.subplots(figsize=(11, 11))

log_Tc = [data[star]["lg(Tc)"][0] for star in data] # K
ax5.plot(log_mass_ZAMS, log_Tc, linewidth=0, color="black", markersize=15, marker="s")

# Fit a los 3 primeros puntos:
pol_1 = np.polyfit(log_mass_ZAMS[:3], log_Tc[:3], 1)
mm_1 = np.linspace(log_mass_ZAMS[0], log_mass_ZAMS[2], 200)
tt_1 = np.polyval(pol_1, mm_1)
ax5.plot(mm_1,tt_1, linewidth=2, color="blue")
ax5.text(log_mass_ZAMS[1]-0.2, log_Tc[1]+0.02, rf"$s_3={pol_1[0]:.2f}$", fontsize=24, color="blue")

# Fit a los 5 siguientes puntos:
pol_2 = np.polyfit(log_mass_ZAMS[3:8], log_Tc[3:8], 1)
mm_2 = np.linspace(log_mass_ZAMS[3], log_mass_ZAMS[7], 200)
tt_2 = np.polyval(pol_2, mm_2)
ax5.plot(mm_2,tt_2, linewidth=2, color="green")
ax5.text(log_mass_ZAMS[5]-0.2, log_Tc[5]+0.02, rf"$s_2={pol_2[0]:.2f}$", fontsize=24, color="green")

# Fit a los 6 últimos puntos:
pol_3 = np.polyfit(log_mass_ZAMS[8:], log_Tc[8:], 1)
mm_3 = np.linspace(log_mass_ZAMS[8], log_mass_ZAMS[-1], 200)
tt_3 = np.polyval(pol_3, mm_3)
ax5.plot(mm_3,tt_3, linewidth=2, color="red")
ax5.text(log_mass_ZAMS[10]-0.2, log_Tc[10]+0.08, rf"$s_1={pol_3[0]:.2f}$", fontsize=24, color="red")

ax5.set_xlabel(r"$\log_{10}(M/\mathrm{M}_{\odot})$", fontsize=30)
ax5.set_ylabel(r"$\log_{10}(T_{{\mathrm{c}}} \, [\unit{\kelvin}])$", fontsize=30)
ax5.tick_params(axis='both', which='major', labelsize=24)
fig5.savefig(os.path.join(plots_dir, "Apartado_2_g.pdf"), format="pdf", bbox_inches='tight')

# Hay que ajustar 3 pendientes distintas: para la opacidad de Kramers hay que considerar 2 valores de n: para pp y para CNO
# 3 #######################################################################################
"""
For the models with M = 1, 3, 9, 40 M⊙:
"""
###########################################################################################

### a) Plot the tracks of the centres of the stars in the (log T, log ρ) plane.
fig6, ax6 = plt.subplots(figsize=(17, 12))
for i, star in enumerate(data):
    if i in [0,3,7,11]:
        length = len(data[star])
        ax6.plot(data[star]["lg(Tc)"], data[star]["lg(rhoc)"], color=colors[i])
        ax6.text(data[star]["lg(Tc)"][0]-0.3, data[star]["lg(rhoc)"][0]-0.22, rf"${star_params["Mass"][i].value}$ $M_{{\odot}}$", color=colors[i], fontsize=24)

# Ajustes del plot:
ax6.set_xlim(6.5, 10)
ax6.set_ylim(-0.5, 10)
ax6.set_xlabel(r"$\log_{10}(T_{{\mathrm{c}}} \, [\unit{\kelvin}])$", fontsize=30)
ax6.set_ylabel(r"$\log_{10}(\rho_{{\mathrm{c}}} \, [\unit{\gram\per\centi\meter\cubed}])$", fontsize=30)

ax6.tick_params(axis='both', which='major', labelsize=24)
### b) Mark the location of the centre of the stars when at the ZAMS.
### c) Mark the location of the centre of the stars when at the TAMS.
for i, star in enumerate(data):
    if i in [0,3,7,11]:
        # ZAMS:
        ax6.plot(data[star]["lg(Tc)"][0], data[star]["lg(rhoc)"][0], marker=">", color="black", 
                markersize=10, linewidth=0, label="ZAMS" if i==0 else None)
        # TAMS:
        ax6.plot(data[star]["lg(Tc)"][compute_TAMS(data[star], 1e-4)], data[star]["lg(rhoc)"][compute_TAMS(data[star], 1e-4)], 
                marker="<", color="black", markersize=10, linewidth=0, label="TAMS" if i==0 else None)
ax6.legend(loc="lower right", fontsize=24)

### d) Plot the areas dominated by a classic ideal gas, a degenerate electron gas,
### a degenerate relativistic electron gas, and a radiation gas. Assume µe = 2
### when dealing with the limit between the classic ideal and the degenerate
### gases and µ = 0.61 when looking for the separation between the classic
### ideal gas and the radiation-dominated region.
mu = 0.61
mu_e = 2
m_H = m_p + m_e
# Gas ideal: P=R rho T / mu. Necesitamos R en unidades de J/K g. Dividimos por la masa molar del H. Llamamos a esta constante R_g
R_g = k_B/m_H
# Gas de electrones no relativistas degenerados: P_e,deg = K1 rho^5/3
K1 = (1/mu_e**(5/3) * (3/np.pi)**(2/3) * h**2 / (20*m_e *m_H**(5/3)))
# Gas de electrones relativistas degenerados: P_e,r-deg = K2 rho^4/3
K2 = (1/mu_e**(4/3) * (3/np.pi)**(1/3) * h * c / (8*m_H**(4/3)))
# Radiación: P_rad = 1/3 a T^4
a = 4*sigma_sb/c

TT = np.linspace(6.5, 10, 1000)
RR = np.linspace(-0.5, 10, 1000)
TEMP, RHO = np.meshgrid(TT, RR)

# Frontera entre gas ideal y radiación: 10 * R_g rho T / mu = 0.1/3 a T^4 --> rho = 0.1/3 a/R_g mu T^3 --> log(rho) = log(0.1/3 a/R_g mu) +3log(T)
F1 = RHO -3*TEMP -np.log10((0.1/3 * a/(R_g) * mu).cgs.value)
# Frontera entre gas ideal y gas degenerado no relativista (e-): K1 rho^5/3 = R_g rho T /mu_e --> rho^2/3 = R_g/(K1*mu_e) T --> log(rho) = 3/2 log(R_g/(K1*mu_e)) + 3/2 log(T)
F2 = RHO -3/2*TEMP -3/2*np.log10((R_g/(K1*mu)).cgs.value)
# Frontera entre gas ideal y gas degenerado relativista (e-): K2 rho^4/3 = R_g rho T /mu_e  --> rho^1/3 = R_g/(K2*mu_e) T --> log(rho) = 3log(R_g/(K2*mu_e)) +3log(T)
F3 = RHO  -3*TEMP -3*np.log10((R_g/(K2*mu)).cgs.value)

# Frontera entre gas degenerado relativista y no relativista: K1 rho^5/3 = K2 rho^4/3 --> rho^1/3 = K2/K1 --> log(rho) = 3*log(K2/K1)
F4 = RHO -3*np.log10((K2/K1).cgs.value)
# Dibujamos las regiones:
Z = np.zeros_like(RHO)
Z[(F1 <= 0)] = 1  
Z[(F1 > 0) & ((F2<0) | (F3<0))] = 2   
Z[((F2>0) & (F3>0) & (F4<0))] = 3
Z[((F2>0) & (F3>0) & (F4>0))] = 4

ax6.contourf(TEMP, RHO, Z, levels=[0, 1, 2, 3, 4], colors="black", alpha=[0.1,0.4,0.6, 0.8])

# Info sobre las regiones en el plot:
ax6.text(9,2,r"\textsc{Radiation Gas}", fontsize=24, color="black")
ax6.text(8.8,6,r"\textsc{Classical Ideal Gas}", fontsize=24, color="black")
ax6.text(7,9,r"\textsc{Degenerate Relativistic Electron Gas}", fontsize=24, color="white")
ax6.text(6.7,5.8,r"\textsc{Degenerate Electron Gas}", fontsize=24, color="white")

# Info sobre las fronteras:
ax6.text(8.2,0.96,r"$P_{\mathrm{rad}} = 10 P_{\mathrm{class}}$",fontsize=20
         ,rotation=np.degrees(np.arctan(3)),rotation_mode='anchor', transform_rotates_text=True)
ax6.text(9,7.45,r"$P_{\mathrm{class}} = P_{e,\mathrm{deg-rel}}$",fontsize=20, color="white"
         ,rotation=np.degrees(np.arctan(3)),rotation_mode='anchor', transform_rotates_text=True)
ax6.text(7,6.7,r"$P_{e,\mathrm{deg}} = P_{e,\mathrm{deg-rel}}$",fontsize=20, color="white"
         ,rotation=0,rotation_mode='anchor', transform_rotates_text=True)

ax6.text(6.7,3.6,r"$P_{\mathrm{class}} = P_{e,\mathrm{deg}}$",fontsize=20, color="white"
         ,rotation=np.degrees(np.arctan(3/2)),rotation_mode='anchor', transform_rotates_text=True)


fig6.savefig(os.path.join(plots_dir, "Apartado_3_abcd.pdf"), format="pdf", bbox_inches='tight')

# 4 #######################################################################################
"""
For the model with M = 4 M⊙:
"""
###########################################################################################

### a) Plot the central abundances of hydrogen, helium, carbon, and oxygen as
### a function of time. Indicate when the star is in the main sequence (MS),
### the Hertszprung gap + RGB phase (RGB), the blue loop (BL), and the
### AGB phase (AGB). You can do that with vertical lines separating the
### evolutionary stages and using labels. Since the MS phase is much longer
### than the others, you can plot a fraction of it only.

### d) Show the evolution of the radius of the star as a function of time. You
### can use the same vertical lines and labels as in the plot for the chemical
### abundances to separate the different evolutionary stages.

# El radio no aparece explícitamente en los datos, así que lo calculamos a partir de la eq. de Stefan-Boltzmann. radio en radios solares:
radius = np.sqrt(10**data["M004Z14V0"]["lg(L)"]) / (10**data["M004Z14V0"]["lg(Teff)"]/5778)**2

fig7, (ax7_1, ax7_2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True)

ax7_1.plot(data["M004Z14V0"]["time"]/1e6, data["M004Z14V0"]["1H_cen"], color="red", label="H")
ax7_1.plot(data["M004Z14V0"]["time"]/1e6, data["M004Z14V0"]["4He_cen"], color="orange", label="He")
ax7_1.plot(data["M004Z14V0"]["time"]/1e6, data["M004Z14V0"]["12C_cen"]+data["M004Z14V0"]["13C_cen"], color="blue", label="C")
ax7_1.plot(data["M004Z14V0"]["time"]/1e6, data["M004Z14V0"]["16O_cen"]+data["M004Z14V0"]["17O_cen"]+data["M004Z14V0"]["18O_cen"], color="green", label="O")

ax7_2.plot(data["M004Z14V0"]["time"]/1e6, radius, color="black")

# Escala para ver con detalle las fases rápidas:
splits = [155, 160, 195]
factors = [1/10, 1/2, 1/10, 1]

inverse_splits = []
inverse_splits.append(splits[0]*factors[0])
inverse_splits.append(splits[0]*factors[0]+ (splits[1]-splits[0])*factors[1])
inverse_splits.append(splits[0]*factors[0]+ (splits[1]-splits[0])*factors[1] + (splits[2]-splits[1])*factors[2])

inverse_factors = [1/x for x in factors]

def forward(x):
    transformed = x.copy()

    mask_1 = (x <= splits[0])
    transformed[mask_1] = x[mask_1] * factors[0]

    mask_2 = (x > splits[0]) & (x <= splits[1])
    transformed[mask_2] = inverse_splits[0] + (x[mask_2] - splits[0]) * factors[1]

    mask_3 = (x > splits[1]) & (x <= splits[2])
    transformed[mask_3] = inverse_splits[1] + (x[mask_3] - splits[1]) *factors[2]

    mask_4 = (x>splits[2])
    transformed[mask_4] = inverse_splits[2] + (x[mask_4] - splits[2]) * factors[3]
        
    return transformed

# Define the inverse transformation
def backward(x):
    original = x.copy()

    mask_1 = (x <= inverse_splits[0])
    original[mask_1] = x[mask_1] * inverse_factors[0]

    mask_2 = (x > inverse_splits[0]) & (x <= inverse_splits[1])
    original[mask_2] = splits[0] + (x[mask_2] - inverse_splits[0]) * inverse_factors[1]

    mask_3 = (x > inverse_splits[1]) & (x <= inverse_splits[2])
    original[mask_3] = splits[1] + (x[mask_3] - inverse_splits[1]) * inverse_factors[2]

    mask_4 = (x>inverse_splits[2])
    original[mask_4] = splits[2] + (x[mask_4] - inverse_splits[2]) * inverse_factors[3]

    return original

#ax7_2.set_xscale('function', functions=(lambda x: x**7, lambda x: x**(1/7)))
ax7_2.set_xscale('function', functions=(forward, backward))

ax7_2.grid(True)
ax7_1.grid(True)
ax7_1.set_xlim(133, np.max(data["M004Z14V0"]["time"]/1e6))
ax7_1.set_ylim(0, 1.2)
ax7_2.set_xticks(np.arange(130, 200, 5))

ax7_2.set_xlabel(r"$t$ $[\unit{\mega\year}]$", fontsize=30)
ax7_2.set_ylabel(r"$R/\mathrm{R_{\odot}}$", fontsize=30)
ax7_1.set_ylabel(r"Central mass fraction", fontsize=30)

ax7_2.tick_params(axis='both', which='major', labelsize=24)
ax7_1.tick_params(axis='both', which='major', labelsize=24)

ax7_1.legend(loc="best", fontsize=24)

# Calculamos las separaciones entre las etapas MS, HG+RGB, BL y AGB:
# TAMS:
TAMS_index = compute_TAMS(data["M009Z14V0"], 1e-4)
ax7_1.axvline(data["M004Z14V0"]["time"][TAMS_index]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)
ax7_2.axvline(data["M004Z14V0"]["time"][TAMS_index]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)
ax7_1.annotate(
        '', 
        xy=(data["M004Z14V0"]["time"][TAMS_index]/1e6, 1.05), 
        xytext=(130, 1.05), 
        arrowprops=dict(arrowstyle='->', color='black', lw=4)
)
ax7_1.text((130 + data["M004Z14V0"]["time"][TAMS_index]/1e6) / 2, 1.1, r'MS', 
             horizontalalignment='center', color='black', fontsize=24)

# BL:
BL_start = compute_BL(data["M009Z14V0"],radius)
ax7_1.axvline(data["M004Z14V0"]["time"][BL_start]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)
ax7_2.axvline(data["M004Z14V0"]["time"][BL_start]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)

ax7_1.axvline(data["M004Z14V0"]["time"][compute_AGB(data["M004Z14V0"], 1e-4)]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)
ax7_1.annotate(
        '', 
        xy=(data["M004Z14V0"]["time"][BL_start]/1e6, 1.05), 
        xytext=(data["M004Z14V0"]["time"][TAMS_index]/1e6, 1.05), 
        arrowprops=dict(arrowstyle='<->', color='black', lw=4)
)
ax7_1.text(155.8, 1.1, r'HG + RGB', 
             horizontalalignment='center', color='black', fontsize=24)


# AGB:
AGB_start = compute_AGB(data["M004Z14V0"], 1e-4)
ax7_1.axvline(data["M004Z14V0"]["time"][AGB_start]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)
ax7_2.axvline(data["M004Z14V0"]["time"][AGB_start]/1e6, color="black",linestyle="dashdot", linewidth=2,
              zorder=0.1, alpha=0.9)

ax7_1.annotate(
        '', 
        xy=(data["M004Z14V0"]["time"][AGB_start]/1e6, 1.05), 
        xytext=(data["M004Z14V0"]["time"][BL_start]/1e6, 1.05), 
        arrowprops=dict(arrowstyle='<->', color='black', lw=4)
)
ax7_1.text(166.3, 1.1, r'BL', 
             horizontalalignment='center', color='black', fontsize=24)

ax7_1.annotate(
        '', 
        xy=(data["M004Z14V0"]["time"][-1]/1e6, 1.05), 
        xytext=(data["M004Z14V0"]["time"][AGB_start]/1e6, 1.05), 
        arrowprops=dict(arrowstyle='<->', color='black', lw=4)
)
ax7_1.text(195.07, 1.1, r'AGB', 
             horizontalalignment='left', color='black', fontsize=24)

fig7.savefig(os.path.join(plots_dir, "Apartado_4_ad.pdf"), format="pdf", bbox_inches='tight')

# Main Sequence acaba cuando se acaba H y He llega al maximo (ZAMS). 
# H. Gap y RGB cuando el radio aumenta y el He sigue cte.
# Blue Loop una vez el núcleo de He se enciende. 
# AGB cuando se acaba el He, C y O se estabilizan y el radio vuelve a aumentar

# En los plots de relaciones de homología, llamar "s" a las pendientes. Plotear 3 pendientes teniendo en cuenta
# ciclo CNO (n=16) y cadena pp (n=4). Para las más calientes (CNO), considerar sólo opacidad de electron scattering, porque la
# opacidad de Kramers no domina a altas temperaturas. Para las pp, considerar electron scattering (masa intermedia) y Kramers para
# las más frías.

### b) Plot the evolutionary track in the HR diagram. Indicate with differ-
### ent colours the main sequence (MS), the Hertszprung gap + RGB phase
### (RGB), the blue loop (BL), and the AGB phase (AGB).
fig8, ax8 = plt.subplots(figsize=(10, 10))

ax8.plot(data["M004Z14V0"]["lg(Teff)"][:TAMS_index+1], data["M004Z14V0"]["lg(L)"][:TAMS_index+1], color="gold", label=r"MS", linewidth=2)
ax8.plot(data["M004Z14V0"]["lg(Teff)"][TAMS_index:BL_start+1], data["M004Z14V0"]["lg(L)"][TAMS_index:BL_start+1], color="orangered", label=r"HG + RGB",linewidth=2)
ax8.plot(data["M004Z14V0"]["lg(Teff)"][BL_start:AGB_start+1], data["M004Z14V0"]["lg(L)"][BL_start:AGB_start+1], color="royalblue", label=r"BL", linewidth=2)
ax8.plot(data["M004Z14V0"]["lg(Teff)"][AGB_start:-1], data["M004Z14V0"]["lg(L)"][AGB_start:-1], color="lime", label=r"AGB", linewidth=2)


# Ajustes del plot:
ax8.invert_xaxis()
ax8.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}} [\unit{\kelvin}])$", fontsize=30)
ax8.set_ylabel(r"$\log_{10}(L/\mathrm{L_{\odot}})$", fontsize=30)
ax8.tick_params(axis='both', which='major', labelsize=24)
ax8.legend(loc="best", fontsize=24)

fig8.savefig(os.path.join(plots_dir, "Apartado_4_b.pdf"), format="pdf", bbox_inches='tight')


### c) Plot the evolution of the centre of the star in the (log T, log ρ) plane. Indi-
### cate with different colours the main sequence (MS), the Hertszprung gap
### + RGB phase (RGB), the blue loop (BL), and the AGB phase (AGB).
fig9, ax9 = plt.subplots(figsize=(10, 10))

ax9.plot(data["M004Z14V0"]["lg(Tc)"][:TAMS_index+1], data["M004Z14V0"]["lg(rhoc)"][:TAMS_index+1], color="gold", label=r"MS", linewidth=2)
ax9.plot(data["M004Z14V0"]["lg(Tc)"][TAMS_index:BL_start+1], data["M004Z14V0"]["lg(rhoc)"][TAMS_index:BL_start+1], color="orangered", label=r"HG + RGB",linewidth=2)
ax9.plot(data["M004Z14V0"]["lg(Tc)"][BL_start:AGB_start+1], data["M004Z14V0"]["lg(rhoc)"][BL_start:AGB_start+1], color="royalblue", label=r"BL", linewidth=2)
ax9.plot(data["M004Z14V0"]["lg(Tc)"][AGB_start:-1], data["M004Z14V0"]["lg(rhoc)"][AGB_start:-1], color="lime", label=r"AGB", linewidth=2)

ax9.set_xlabel(r"$\log_{10}(T_{{\mathrm{c}}} \, [\unit{\kelvin}])$", fontsize=30)
ax9.set_ylabel(r"$\log_{10}(\rho_{{\mathrm{c}}} \, [\unit{\gram\per\centi\meter\cubed}])$", fontsize=30)
ax9.tick_params(axis='both', which='major', labelsize=24)
ax9.legend(loc="lower right", fontsize=24)

TT = np.linspace(ax9.get_xlim()[0], ax9.get_xlim()[1], 2000)
RR = np.linspace(ax9.get_ylim()[0], ax9.get_ylim()[1], 2000)
TEMP, RHO = np.meshgrid(TT, RR)

# Frontera entre gas ideal y radiación: 10 * R_g rho T / mu = 0.1/3 a T^4 --> rho = 0.1/3 a/R_g mu T^3 --> log(rho) = log(0.1/3 a/R_g mu) +3log(T)
F1 = RHO -3*TEMP -np.log10((0.1/3 * a/(R_g) * mu).cgs.value)
# Frontera entre gas ideal y gas degenerado no relativista (e-): K1 rho^5/3 = R_g rho T /mu_e --> rho^2/3 = R_g/(K1*mu_e) T --> log(rho) = 3/2 log(R_g/(K1*mu_e)) + 3/2 log(T)
F2 = RHO -3/2*TEMP -3/2*np.log10((R_g/(K1*mu)).cgs.value)
# Frontera entre gas ideal y gas degenerado relativista (e-): K2 rho^4/3 = R_g rho T /mu_e  --> rho^1/3 = R_g/(K2*mu_e) T --> log(rho) = 3log(R_g/(K2*mu_e)) +3log(T)
F3 = RHO  -3*TEMP -3*np.log10((R_g/(K2*mu)).cgs.value)

# Frontera entre gas degenerado relativista y no relativista: K1 rho^5/3 = K2 rho^4/3 --> rho^1/3 = K2/K1 --> log(rho) = 3*log(K2/K1)
F4 = RHO -3*np.log10((K2/K1).cgs.value)
# Dibujamos las regiones:
Z = np.zeros_like(RHO)
Z[(F1 <= 0)] = 1  
Z[(F1 > 0) & ((F2<0) | (F3<0))] = 2   
Z[((F2>0) & (F3>0) & (F4<0))] = 3
Z[((F2>0) & (F3>0) & (F4>0))] = 4

ax9.contourf(TEMP, RHO, Z, levels=[0, 1, 2, 3, 4], colors="black", alpha=[0.1,0.4,0.6, 0.8])

fig9.savefig(os.path.join(plots_dir, "Apartado_4_c.pdf"), format="pdf", bbox_inches='tight')
