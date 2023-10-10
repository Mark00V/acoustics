import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance
from numpy import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

#==================================================
# Allgemeine Definitionen
freq = 400
c = 340
rohair=1.21
pos_quelle = (0.0, 0.0)
Zi=(rohair*c)
impwert=1/Zi
#--------------------------------------------------
omega=freq*2*math.pi
cair=c
#==================================================

#==================================================
#Notwendige Funktionen und Definitionen für Berechnung Elementmatrizen p=1

#--------------------------------------------------
# Numerische Integration
numintgld1=np.array([[0.21132486,0.78867513],[0.50000000,0.50000000]]) # Numerische Integration 1D Knoten und Gewichte
numintgld2nodes=np.array([[0.21132487, 0.21132487], [0.21132487, 0.78867513], \
                          [0.78867513, 0.21132487], [0.78867513, 0.78867513]])  # Numerische Integration 2D Knoten
numintgld2weights=np.array([0.25000000, 0.25000000, 0.25000000, 0.25000000])      # Numerische Integration 2D Gewichte
#--------------------------------------------------
# Determinante der Jacobimatrix für die Eckpunkte {{x1, y1}, {x2, y2}, {x3, y3}, {x4,y4}}
# -> {{ul}, {ur}, {ol}, {or}} mit den Funktionswerten xi1 und xi2
def jacdetc(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4):
    f=0.0-1.0*x2*y1+1.0*x3*y1-1.0*x3*xi1*y1+1.0*x4*xi1*y1+ \
       1.0*x2*xi2*y1-1.0*x4*xi2*y1+1.0*x1*y2-1.0*x3*y2+1.0*x3*xi1*y2- \
       1.0*x4*xi1*y2-1.0*x1*xi2*y2+1.0*x3*xi2*y2-1.0*x1*y3+1.0*x2*y3+ \
       1.0*x1*xi1*y3-1.0*x2*xi1*y3-1.0*x2*xi2*y3+1.0*x4*xi2*y3- \
       1.0*x1*xi1*y4+1.0*x2*xi1*y4+1.0*x1*xi2*y4-1.0*x3*xi2*y4
    return f
#--------------------------------------------------
# Formfunktionen philequdist und phiqequdist
def philequdist(knoten,xi):
    knoten=int(knoten)
    if knoten==1:
        f=1.0 - xi
    elif knoten==2:
        f=xi
    return f

def phiqequdist(knoten,xi1,xi2):
    knoten=int(knoten)
    if knoten==1:
        f=(1 - xi1)*(1 - xi2)
    elif knoten==2:
        f=xi1*(1 - xi2)
    elif knoten==3:
        f=(1 - xi1)*xi2
    elif knoten==4:
        f=xi1*xi2
    return f

def phiqequdistarray(xi1,xi2):
    f=np.array([[(1 - xi1)*(1 - xi2)],[xi1*(1 - xi2)],[(1 - xi1)*xi2],[xi1*xi2]])
    return f

#--------------------------------------------------
# phiqequdistgrad
def phiqequdistgrad(knoten,xi1,xi2):
    knoten=int(knoten)   
    if knoten==1:
        f=np.array([-1 + xi2, -1 + xi1])
    elif knoten==2:
        f=np.array([1 - xi2, -xi1])
    elif knoten==3:
        f=np.array([-xi2, 1 - xi1])
    elif knoten==4:
        f=np.array([xi2, xi1])
    return f


#--------------------------------------------------
# gradmatequdist
def gradmatequdist(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4):
    jacinvtrans=np.array([[(-1.0*(1.0-1.0*xi1)*y1-1.0*xi1*y2+(1.0-1.0*xi1)*y3+xi1*y4)/(0.0- \
        1.0*x2*y1+1.0*x3*y1-1.0*x3*xi1*y1+1.0*x4*xi1*y1+ \
        1.0*x2*xi2*y1-1.0*x4*xi2*y1+1.0*x1*y2-1.0*x3*y2+ \
        1.0*x3*xi1*y2-1.0*x4*xi1*y2-1.0*x1*xi2*y2+1.0*x3*xi2*y2- \
        1.0*x1*y3+1.0*x2*y3+1.0*x1*xi1*y3-1.0*x2*xi1*y3- \
        1.0*x2*xi2*y3+1.0*x4*xi2*y3-1.0*x1*xi1*y4+1.0*x2*xi1*y4+ \
        1.0*x1*xi2*y4-1.0*x3*xi2*y4), \
        (1.0*(1.0-1.0*xi2)*y1-(1.0-1.0*xi2)*y2+ \
        1.0*xi2*y3-xi2*y4)/(0.0-1.0*x2*y1+1.0*x3*y1-1.0*x3*xi1*y1+ \
        1.0*x4*xi1*y1+1.0*x2*xi2*y1-1.0*x4*xi2*y1+1.0*x1*y2- \
        1.0*x3*y2+1.0*x3*xi1*y2-1.0*x4*xi1*y2-1.0*x1*xi2*y2+ \
        1.0*x3*xi2*y2-1.0*x1*y3+1.0*x2*y3+1.0*x1*xi1*y3- \
        1.0*x2*xi1*y3-1.0*x2*xi2*y3+1.0*x4*xi2*y3-1.0*x1*xi1*y4+ \
        1.0*x2*xi1*y4+1.0*x1*xi2*y4-1.0*x3*xi2*y4)], \
        [(1.0*x1*(1.0-1.0*xi1)-x3*(1.0-1.0*xi1)+1.0*x2*xi1-x4*xi1)/(0.0- \
        1.0*x2*y1+1.0*x3*y1-1.0*x3*xi1*y1+1.0*x4*xi1*y1+ \
        1.0*x2*xi2*y1-1.0*x4*xi2*y1+1.0*x1*y2-1.0*x3*y2+ \
        1.0*x3*xi1*y2-1.0*x4*xi1*y2-1.0*x1*xi2*y2+1.0*x3*xi2*y2- \
        1.0*x1*y3+1.0*x2*y3+1.0*x1*xi1*y3-1.0*x2*xi1*y3- \
        1.0*x2*xi2*y3+1.0*x4*xi2*y3-1.0*x1*xi1*y4+1.0*x2*xi1*y4+ \
        1.0*x1*xi2*y4-1.0*x3*xi2*y4), \
        (-1.0*x1*(1.0-1.0*xi2)+ \
        x2*(1.0-1.0*xi2)-1.0*x3*xi2+x4*xi2)/(0.0-1.0*x2*y1+ \
        1.0*x3*y1-1.0*x3*xi1*y1+1.0*x4*xi1*y1+1.0*x2*xi2*y1- \
        1.0*x4*xi2*y1+1.0*x1*y2-1.0*x3*y2+1.0*x3*xi1*y2- \
        1.0*x4*xi1*y2-1.0*x1*xi2*y2+1.0*x3*xi2*y2-1.0*x1*y3+ \
        1.0*x2*y3+1.0*x1*xi1*y3-1.0*x2*xi1*y3-1.0*x2*xi2*y3+ \
        1.0*x4*xi2*y3-1.0*x1*xi1*y4+1.0*x2*xi1*y4+1.0*x1*xi2*y4- \
        1.0*x3*xi2*y4)]])
    
    vekgradmat=np.zeros((4,2))
    
    for knoten in range(4):
        matrow=jacinvtrans@phiqequdistgrad(knoten+1,xi1,xi2)
        vekgradmat[knoten]=matrow
    return vekgradmat
#--------------------------------------------------
#==================================================


#--------------------------------------------------
# Import Meshdata
# elementindex_schallwand_fein.csv bis ca. 300 Hz ohne gravierende numerische Fehler
elementindex=np.loadtxt("elementindex_schallwand.csv",delimiter=" ", dtype=str)
elementindex=elementindex[:,0:4]
elementindex=elementindex.astype(np.uint32)

pointcoordinates=np.loadtxt("pointcoordinates_schallwand.csv",delimiter=" ", dtype=str)
pointcoordinates=pointcoordinates[:,0:2]
pointcoordinates=pointcoordinates.astype(np.float32)

maxelement=len(elementindex)
maxnode=len(pointcoordinates)
zuordtab=elementindex
allnodes=np.zeros((maxnode,3))

k=1
for i in range(maxnode):
    allnodes[i,0]=pointcoordinates[i,0]
    allnodes[i,1]=pointcoordinates[i,1]
    allnodes[i,2]=int(k)
    k+=1

#--------------------------------------------------
# Boundaries
minxpc=min(allnodes[:,0])
maxxpc=max(allnodes[:,0])
minypc=min(allnodes[:,1])
maxypc=max(allnodes[:,1])

boundaryleft=allnodes[allnodes[:,0]==minxpc]
boundaryright=allnodes[allnodes[:,0]==maxxpc]
boundarybottom=allnodes[allnodes[:,1]==minypc]
boundarytop=allnodes[allnodes[:,1]==maxypc]

#--------------------------------------------------
# Randbedingungen Impedanz, Beachte Indexierung hier gegenüber struct mesh (allnods[:,2] startet hier mit 1!
boundaryleftn=boundaryleft[:,2]-1
boundaryrightn=boundaryright[:,2]-1
boundarybottomn=boundarybottom[:,2]-1
boundarytopn=boundarytop[:,2]-1

zuordtableft_ibc=[boundaryleftn[i:i+2] for i in range(0,len(boundaryleftn)-1,1)]
zuordtabright_ibc=[boundaryrightn[i:i+2] for i in range(0,len(boundaryrightn)-1,1)]
zuordtablebottom_ibc=[boundarybottomn[i:i+2] for i in range(0,len(boundarybottomn)-1,1)]
zuordtabtop_ibc=[boundarytopn[i:i+2] for i in range(0,len(boundarytopn)-1,1)]

#--------------------------------------------------
# Erstelle Elementmatrizen
intnodes=numintgld2nodes
intweights=numintgld2weights
allelementssteif=np.zeros((maxelement,4,4))
allelementsmass=np.zeros((maxelement,4,4))

for ie in range(maxelement):
    elesteifmat=np.zeros((4,4))
    elemassenmat=np.zeros((4,4))
    
    ielem=zuordtab[ie]
    knoten1=ielem[0]
    knoten2=ielem[1]
    knoten3=ielem[2]
    knoten4=ielem[3]
    x1=pointcoordinates[knoten1,0]
    y1=pointcoordinates[knoten1,1]
    x2=pointcoordinates[knoten2,0]
    y2=pointcoordinates[knoten2,1]
    x3=pointcoordinates[knoten3,0]
    y3=pointcoordinates[knoten3,1]
    x4=pointcoordinates[knoten4,0]
    y4=pointcoordinates[knoten4,1]
    
    for i in range(4):
        xi1=intnodes[i,0]
        xi2=intnodes[i,1]
        gr=gradmatequdist(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4)
        grt=np.transpose(gr)
        grxgrt=gr@grt
        jd=jacdetc(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4)
        fp=grxgrt*jd*intweights
        elesteifmat=elesteifmat+fp
    elesteifmat=elesteifmat
    
    for i in range(4):
        xi1=intnodes[i,0]
        xi2=intnodes[i,1]
        phi=phiqequdistarray(xi1,xi2)
        phit=np.transpose(phiqequdistarray(xi1,xi2))
        phixphit=phi@phit
        jd=jacdetc(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4)
        fp=phixphit*jd*intweights
        elemassenmat=elemassenmat+fp
    
    allelementssteif[ie]=elesteifmat
    allelementsmass[ie]=elemassenmat

#--------------------------------------------------
# Erstelle Randelementematrizen Impedanz

intnodes = numintgld1[0]
intweights = numintgld1[1]

# links
neumannwert=impwert
allimpelementslinks=np.zeros((len(zuordtableft_ibc),2,2))
for ielem in range(0,len(zuordtableft_ibc)):
    impmatele=np.zeros((2,2))
    nb=np.zeros((2,2))
    element=zuordtableft_ibc[ielem]
    knoten1=int(element[0])
    knoten2=int(element[1])
    pos_y_knoten1=allnodes[knoten1,1]
    pos_y_knoten2=allnodes[knoten2,1]
    length=abs(pos_y_knoten2-pos_y_knoten1)
    for j in range(0,2):
        for i in range(0,2):
            for ii in range(0,2):
                val=philequdist(i+1,intnodes[j])*philequdist(ii+1,intnodes[j])*intweights[j]
                nb[i,ii]=val
        impmatele=impmatele+nb
    impmatele=impmatele*length*neumannwert   
    allimpelementslinks[ielem]=impmatele

# oben
neumannwert=impwert
allimpelementsoben=np.zeros((len(zuordtabtop_ibc),2,2))
for ielem in range(0,len(zuordtabtop_ibc)):
    impmatele=np.zeros((2,2))
    nb=np.zeros((2,2))
    element=zuordtabtop_ibc[ielem]
    knoten1=int(element[0])
    knoten2=int(element[1])
    pos_x_knoten1=allnodes[knoten1,0]
    pos_x_knoten2=allnodes[knoten2,0]
    length=abs(pos_y_knoten2-pos_y_knoten1)
    for j in range(0,2):
        for i in range(0,2):
            for ii in range(0,2):
                val=philequdist(i+1,intnodes[j])*philequdist(ii+1,intnodes[j])*intweights[j]
                nb[i,ii]=val
        impmatele=impmatele+nb
    impmatele=impmatele*length*neumannwert   
    allimpelementsoben[ielem]=impmatele

# rechts
neumannwert=impwert
allimpelementsrechts=np.zeros((len(zuordtabright_ibc),2,2))
for ielem in range(0,len(zuordtabright_ibc)):
    impmatele=np.zeros((2,2))
    nb=np.zeros((2,2))
    element=zuordtabright_ibc[ielem]
    knoten1=int(element[0])
    knoten2=int(element[1])
    pos_y_knoten1=allnodes[knoten1,1]
    pos_y_knoten2=allnodes[knoten2,1]
    length=abs(pos_y_knoten2-pos_y_knoten1)
    for j in range(0,2):
        for i in range(0,2):
            for ii in range(0,2):
                val=philequdist(i+1,intnodes[j])*philequdist(ii+1,intnodes[j])*intweights[j]
                nb[i,ii]=val
        impmatele=impmatele+nb
    impmatele=impmatele*length*neumannwert   
    allimpelementsrechts[ielem]=impmatele

#--------------------------------------------------
# Erstelle Impedanzmatrix

# links
sysimparraylinks=np.zeros((maxnode,maxnode))
for ielem in range(len(zuordtableft_ibc)):
    eleimpmat=allimpelementslinks[ielem]
    for a in range(2):
        for b in range(2):
            zhab=zuordtableft_ibc[ielem]
            zta=int(zhab[a])
            ztb=int(zhab[b])
            sysimparraylinks[zta,ztb]=sysimparraylinks[zta,ztb]+eleimpmat[a,b]

# oben
sysimparrayoben=np.zeros((maxnode,maxnode))
for ielem in range(len(zuordtabtop_ibc)):
    eleimpmat=allimpelementsoben[ielem]
    for a in range(2):
        for b in range(2):
            zhab=zuordtabtop_ibc[ielem]
            zta=int(zhab[a])
            ztb=int(zhab[b])
            sysimparrayoben[zta,ztb]=sysimparrayoben[zta,ztb]+eleimpmat[a,b]

# rechts
sysimparrayrechts=np.zeros((maxnode,maxnode))
for ielem in range(len(zuordtabright_ibc)):
    eleimpmat=allimpelementsrechts[ielem]
    for a in range(2):
        for b in range(2):
            zhab=zuordtabright_ibc[ielem]
            zta=int(zhab[a])
            ztb=int(zhab[b])
            sysimparrayrechts[zta,ztb]=sysimparrayrechts[zta,ztb]+eleimpmat[a,b]

sysimparray=sysimparraylinks+sysimparrayoben+sysimparrayrechts

#--------------------------------------------------
# Erstelle Systemmatrizen

syssteifarray=np.zeros((maxnode,maxnode))
sysmassarray=np.zeros((maxnode,maxnode))

for ielem in range(maxelement):
    elesteifmat=allelementssteif[ielem]
    elemassenmat=allelementsmass[ielem]
    for a in range(4):
        for b in range(4):
            zta=int(zuordtab[ielem,a])
            ztb=int(zuordtab[ielem,b])
            syssteifarray[zta,ztb]=syssteifarray[zta,ztb]+elesteifmat[a,b]
            sysmassarray[zta,ztb]=sysmassarray[zta,ztb]+elemassenmat[a,b]

syssteif = syssteifarray
sysmass = sysmassarray

sysmatfreq = (1/rohair)*syssteif - (1/rohair)*(1/cair**2)*(omega**2)*sysmass+omega*sysimparray*1j
sysmatfreqh = (1/rohair)*syssteif - (1/rohair)*(1/cair**2)*(omega**2)*sysmass+omega*sysimparray*1j # WARUM IST DAS NOTWENDIG???? Wieso werden die Werte nicht einfach neu zugewiesen?


#--------------------------------------------------
# Quelle Monopolquelle und Einbau in Lastvektor
# Position Monopolquelle hier x=0.1,y=0.2
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

cn_posquelle=closest_node(pos_quelle, allnodes[:,0:2])
cn_allnodes=allnodes[(allnodes[:,0]==cn_posquelle[0])&(allnodes[:,1]==cn_posquelle[1])]

# Einbau Monopolquelle
posknoten=int(cn_allnodes[0,2])
mpsource = 4*math.pi/rohair*((2*rohair*omega)/((2*math.pi)**2))**0.5
source=np.array([posknoten,mpsource])

lastvektor=np.zeros(maxnode)
lastvektor[int(source[0])]=lastvektor[int(source[0])]+source[1]

#--------------------------------------------------
# Einarbeitung übriger Randbedingungen
# Hier nicht nötig, da Neumann
sysmatfreqred=sysmatfreq

#--------------------------------------------------
# Lösung GLS
solution=np.linalg.solve(sysmatfreqred, lastvektor)

solutionallnodes=np.zeros((maxnode,3))
for i in range(maxnode):
    solutionallnodes[i,0]=allnodes[i,0]
    solutionallnodes[i,1]=allnodes[i,1]

#--------------------------------------------------
# Berechnung Schalldrucklevel
pref=20*10**(-6)
solutiondruck=solution # Druck (mit Im-Teil)
solutionspl=abs(20*np.log10(solution/pref)) # Schalldrucklevel

#--------------------------------------------------
# Crop data
def datacropped(data,minvalue,maxvalue):
    lendata=len(data)
    newdata=np.zeros(lendata)
    for i in range(lendata):
        newdata[i]=data[i]
        if data[i]<minvalue:
            newdata[i]=minvalue
        if data[i]>maxvalue:
            newdata[i]=maxvalue

    return newdata

newspldata=datacropped(solutionspl,40,160)
#--------------------------------------------------
# Ausgabe Contourplot
datax=solutionallnodes[:,0]
datay=solutionallnodes[:,1]
dataz=np.real(newspldata)

minx=min(datax)
maxx=max(datax)
miny=min(datay)
maxy=max(datay)

points=solutionallnodes[:,(0,1)]
values=dataz

grid_x, grid_y = np.mgrid[minx:maxx:600j, miny:maxy:600j]

grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

# Contourplot
nr_of_contours=100 # Contouren insgesamt, hoher Wert für Quasi-Densityplot
nr_of_contourlines=5 # EIngezeichnete Contourlinien, Wert nicht exakt...
aspectxy=1
ctlines=int(nr_of_contours/nr_of_contourlines)

dataX=grid_x
dataY=grid_y
dataZ=grid_z1

fig1, ax = plt.subplots()
CS1 = ax.contourf(dataX, dataY, dataZ, nr_of_contours, cmap=plt.cm.gnuplot2)
ax.set_title('Druckfeld')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect(aspectxy)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = fig1.colorbar(CS1,cax=cax)
cbar.ax.set_ylabel('Druck [p]')

#CS2 = ax.contour(CS1, levels=CS1.levels[::ctlines], colors='gray',linestyles="dashed")

#cbar.add_lines(CS2)
plt.show()

# IMplot
##plt.imshow(grid_z1.T, extent=(minx,maxx,miny,maxy), origin='lower')
##plt.gcf().set_size_inches(6, 6)
##plt.show()
