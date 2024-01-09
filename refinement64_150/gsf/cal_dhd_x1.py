#from cal_pdb_feature import *
from Bio.PDB import calc_dihedral
import numpy as np

def getDihedral(p):
    ''' Reference: http://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python'''
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def getAngle(angle):
    ''' Convert -pi ~ pi to 0 to 2*pi '''
    if angle < 0:
        angle = 360 + angle;
    return round(angle,4);
def getResidueDict(residue):
    ''' Convert residue to a dict, key is atom name and value is coordinates '''
    rDict = {};
    atoms = list(residue.get_atoms());
    for atom in atoms:
        rDict[atom.get_id()] = np.array(atom.get_coord());
    return rDict;

def getChi(atomList):
    ''' Get all chi angles of the side-chain of a residue '''
    chi = []
    if len(atomList) < 4:
        return chi
    for i in range(len(atomList)-3):
        chi.append(getAngle(getDihedral([atomList[i], atomList[i+1], atomList[i+2], atomList[i+3]])));
    return chi
def cal_dhd_x1(res):
    angles = []
    currAtoms = getResidueDict(res)
    resName = res.resname
    atomList =[]
    if resName == 'ARG': # X1, X2, X3, X4
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
        atomList.append(currAtoms['CD']);
        atomList.append(currAtoms['NE']);
        atomList.append(currAtoms['CZ']);
    if resName == 'ASN': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'ASP': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'CYS': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['SG']);
    if resName == 'GLU': # X1, X2
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
        atomList.append(currAtoms['CD']);
    if resName == 'GLN': # X1, X2
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
        atomList.append(currAtoms['CD']);
    if resName == 'HIS': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'ILE': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG1']);
    if resName == 'LEU': # X1, X2
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
        atomList.append(currAtoms['CD1']);
    if resName == 'LYS': # X1, X2, X3, X4
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
        atomList.append(currAtoms['CD']);
        atomList.append(currAtoms['CE']);
        atomList.append(currAtoms['NZ']);
    if resName == 'MET': # X1, X2, X3
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
        atomList.append(currAtoms['SD']);
        atomList.append(currAtoms['CE']);
    if resName == 'PHE': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'PRO': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'SER': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['OG']);
    if resName == 'THR': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['OG1']);
    if resName == 'TRP': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'TYR': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG']);
    if resName == 'VAL': # X1
        atomList.append(currAtoms['N']);
        atomList.append(currAtoms['CA']);
        atomList.append(currAtoms['CB']);
        atomList.append(currAtoms['CG1']);
    #print(atomList)
    #angles = angles + getChi(atomList)
    angles = angles + getChi(atomList)
    return angles 
def dhd_x1_angle(aa_list_full):
    dhd_x1 = []
    for i in range(len(aa_list_full)):
        try:
            angles= cal_dhd_x1(aa_list_full[i])
        except:
            angles = []
        if not angles:
            dhd_x1.append(None)
        else:
            dhd_x1.append(angles[0])
    return dhd_x1

     
       
            

# pdb_list_file = "/state/partition1/cxy/cullpdb_pc50_res2.0_R0.25_d191010_chains17543"
# pdb_id, pdb_chain = get_id_chain_name(pdb_list_file)
# num = list(range(len(pdb_id)))
# #print(len(num))
# for i in tqdm(num[:2]):  #n is the number of subprocesses
#     print(pdb_id[i])
#     if len(pdb_id[i]) !=4:
#         continue
#     #pdb_name=path + "pdb"+pdb_id[i].lower()+'.ent'
#     pdb_name = path+pdb_id[i].lower()+".pdb.gz"
#     chain = pdb_chain[i]

#     #print(pdb_name)
#     #print("reading %s..." % pdb_name)
#     try:
#         aa_list_full = read_pdb(pdb_name, chain)
#         #ss = get_dssp(pdb_name,chain)
#     except:
#         print("read %s fail " % pdb_name)
#         continue
#     ca_list =get_atom_list_npy(aa_list_full,'CA')
#     ca_dist = cal_dist(ca_list)
#     for i in range(len(ca_dist)):
#         print(aa_list_full[i])
#         atomList,angles= cal_dhd_x1(aa_list_full[i])
