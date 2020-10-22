import torch
from NCC import NCC
import cv2
import numpy as np
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt

npImg1 = cv2.imread("images/knee1.bmp")
npImg2 = cv2.imread("images/knee2.bmp")
img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float()/255.0
img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0)/255.0

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()
img1 = Variable(img1, requires_grad=False)
img2 = Variable(img2, requires_grad=True)

ncc = NCC()
optimizer = optim.Adam([img2], lr=0.01)
a = 0

while a <= 20:
    loss=0.0
    optimizer.zero_grad()
    ncc_response = ncc(img1, img2)
    i1 = img1.permute(1,2,0).cpu().numpy()
    i2 = torch.squeeze(img2, 0).permute(1,2,0).cpu().detach().numpy()

    if(a == 0 or a==20):
      fig=plt.figure()
      fig.add_subplot(1,2,1)
      plt.imshow(i1,cmap="gray")
      plt.title("Fixed Image")
      fig.add_subplot(1,2,2)
      plt.imshow(i2,cmap="gray")
      plt.title("Moving Image")
      plt.show()

    loss = loss - ncc_response.max()
    loss.backward()
    optimizer.step()

    print("Interation: ", a)
    print(ncc_response.max())
    print(loss)
    a +=1