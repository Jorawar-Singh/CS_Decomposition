# Python3 code for the algorithm presented in the paper titled "A linear optical scheme to implement positive operator valued measures"
# author: Jorawar Singh
# If you use this code, please consider citing the article.


# Helper Libraries
import numpy as np

# Function definitions
def getS(m):
    """
    Returns the sine matrix corresponding to the input cosine matrix

    Inputs
      m - [np.matrix] the cosine matrix
    
    Outputs
      np.matrix
    """
    return np.matrix([[np.sqrt(1-m[0]**2),0],[0,np.sqrt(1-m[1]**2)]])

def isUnitary(m):
    """
    Checks whether the given matrix is Unitary (precision upto 10th decimal place).

    Inputs
      m - [np.matrix] matrix
    
    Outputs
      Boolean
    """
    return np.all(np.dot(m,m.getH())-np.identity(2) < 10**-10)

def decompose(k):
    """
    Returns 4 lists of left decomposed, cosine, sine, and right-decomposed matrix corresponding to the input kraus operators

    Inputs
      k         - [list] kraus operators

    Outputs
      list
    """

    # Calculating Control Unitaries using SVD
    # L_m_1 - left control unitary
    # C_m_1 - cosine/diagonal matrix
    # S_m_1 - sine/diagonal matrix
    # R_m_1 - right control unitary
    L_m_1, C_m_1, R_m_1 = np.linalg.svd(kraus_0)
    S_m_1 = getS(C_m_1)
    C_m_1 = np.matrix([[C_m_1[0],0],[0,C_m_1[1]]])  #converting the list of eigenvalues into a diagonal matrix

    # For the second step, we right multiply the inverse of R_m_1 and S_m_1 with kraus_1 and then calculate the control unitaries of the resulting matrix using SVD
    kraus_1_bar = np.dot(np.dot(kraus_1,np.linalg.inv(R_m_1)),np.linalg.inv(S_m_1))
    L_m_2, C_m_2, R_m_2 = np.linalg.svd(kraus_1_bar)
    S_m_2 = getS(C_m_2)
    C_m_2 = np.matrix([[C_m_2[0],0],[0,C_m_2[1]]])

    # Similarly for R_m_2, S_m_2 and kraus_2
    kraus_2_bar = np.dot(np.dot(np.dot(np.dot(kraus_2,np.linalg.inv(R_m_1)), np.linalg.inv(S_m_1)), np.linalg.inv(R_m_2)), np.linalg.inv(S_m_2))
    L_m_3, C_m_3, R_m_3 = np.linalg.svd(kraus_2_bar)
    S_m_3 = getS(C_m_3)
    C_m_3 = np.matrix([[C_m_3[0],0],[0,C_m_3[1]]])

    # Similarly for R_m_3, S_m_3 and kraus_3
    kraus_3_bar = np.dot(np.dot(np.dot(np.dot(np.dot(kraus_3, np.linalg.inv(R_m_1)), np.linalg.inv(S_m_1)), np.linalg.inv(R_m_2)), np.linalg.inv(S_m_2)),np.linalg.inv(R_m_3))
    L_m_4 = kraus_3_bar

    return [L_m_1,L_m_2,L_m_3,L_m_4],[C_m_1,C_m_2,C_m_3],[S_m_1,S_m_2,S_m_3],[R_m_1,R_m_2,R_m_3]

##############################################################################

# Constants
r2 = np.sqrt(2)
a = np.exp(2*np.pi*1j/3)
am = np.exp(-2*np.pi*1j/3)
a2 = np.exp(4*np.pi*1j/3)
a2m = np.exp(-4*np.pi*1j/3)

# Defining the Kraus Operators corresponding to SIC-POVMS
kraus_0 = np.matrix([[1/r2,0],[0,0]])
kraus_1 = np.matrix([[1/r2,1],[1,r2]])/3
kraus_2 = np.matrix([[1/r2,am],[a,r2]])/3
kraus_3 = np.matrix([[1/r2,a2m],[a2,r2]])/3

# Decomposing into Left, Cosine, Sine, and Right matrices
L,C,S,R = decompose([kraus_0, kraus_1, kraus_2, kraus_3])

# Printing the various matrices
print("L matrices")
print("L1:\n",L[0],"\n")
print("L2:\n",L[1],"\n")
print("L3:\n",L[2],"\n")
print("L4:\n",L[3],"\n\n")
print("C matrices")
print("C1:\n",C[0],"\n")
print("C2:\n",C[1],"\n")
print("C3:\n",C[2],"\n\n")
print("S matirces")
print("S1:\n",S[0],"\n")
print("S2:\n",S[1],"\n")
print("S3:\n",S[2],"\n\n")
print("R matrices")
print("R1:\n",R[0],"\n")
print("R2:\n",R[1],"\n")
print("R3:\n",R[2],"\n\n")
