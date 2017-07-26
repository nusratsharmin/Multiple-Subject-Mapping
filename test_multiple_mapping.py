import numpy as np
from dipy.tracking.distances import bundles_distances_mam
from dipy.io.pickles import load_pickle
from nibabel import trackvis
from dipy.viz import fvtk



def load_tractography(filename_tractography ):
    
  
    
    tractography, header = trackvis.read(filename_tractography)
    tractography = [streamline[0] for streamline in tractography]
    return  tractography


def compute_partial_cost(A, B, assignment):
  swap = False
  if len(A) > len(B):
      A, B = B, A
      swap = True

  costAB = np.zeros(len(A))
  for i, idx in enumerate(assignment):
      costAB[i] = bundles_distances_mam([A[i]], [B[idx]], metric='avg') / min(len(A[i]), len(B[idx]))

  cost01 = costAB
  if swap:
      cost01 = -np.ones(len(B))
      cost01[assignment] = costAB
      cost01[cost01 == -1] = costAB.max()

  return cost01


if __name__ == '__main__':
    
    
  subject_id = ["100307", "124422", "161731", "199655", "201111", "239944", "245333", "366446", "528446", "856766"] 
  
  sidA = "245333"
  filename_tractographyA = '/home/nusrat/HCP/MICCAI2015_DTI_EUDX/'+ sidA +'_1M_apss.trk'
  tractographyA = np.array(load_tractography(filename_tractographyA))
  subject_id.remove(sidA)
  cost = 0.0
  for sidB in subject_id:
      print(sidB)
      filename_tractographyB = '/home/nusrat/HCP/MICCAI2015_DTI_EUDX/'+ sidB +'_1M_apss.trk'
      tractographyB = np.array(load_tractography(filename_tractographyB))
      name = sidB + sidA
      if len(tractographyA) > len(tractographyB):
          name = sidA + sidB
          
      assignment = load_pickle('result_wb_solp/Assignment/'+ name +'.pkl')
      cost += compute_partial_cost(tractographyA, tractographyB, assignment)
  
  rank = cost.argsort()
  #plt.hist(cost, bins=50)
  #plt.show()
  
  show=True  
  if show: 
   ren = fvtk.ren()           
   #fvtk.add(ren, fvtk.line(tractography_1[rank[:100]].tolist(), fvtk.colors.white, linewidth=2, opacity=0.5))
   #fvtk.add(ren, fvtk.line(tractography_1[rank[-100:]].tolist(), fvtk.colors.blue, linewidth=2, opacity=0.5))
   fvtk.add(ren, fvtk.line(tractographyA[rank[:2000]].tolist(), fvtk.colors.white, linewidth=2, opacity=0.5))
   fvtk.add(ren, fvtk.line(tractographyA[rank[-2000:]].tolist(), fvtk.colors.blue, linewidth=2, opacity=0.5))
   fvtk.show(ren)  
   
  
