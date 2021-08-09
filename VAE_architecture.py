#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:27:05 2020

@author: zhangjr
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import random
import matplotlib.pyplot as plt

#gpu=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self,x_dim,h_dim1,h_dim2,z_dim):
        super(VAE,self).__init__()
        
        #encoder
        self.fc1=nn.Linear(x_dim,h_dim1)
        self.fc2=nn.Linear(h_dim1,h_dim2)
        self.fc31=nn.Linear(h_dim2,z_dim)
        self.fc32=nn.Linear(h_dim2,z_dim)

        #decoder
        self.fc4=nn.Linear(z_dim,h_dim2)
        self.fc5=nn.Linear(h_dim2,h_dim1)
        self.fc6=nn.Linear(h_dim1,x_dim)
        
    def encoder(self,x):
        h=F.relu(self.fc1(x))
        h=F.relu(self.fc2(h))
        
        return self.fc31(h),self.fc32(h) #mu,log_var

    def sampling(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std)

        return mu+eps*std
    
    def decoder(self,z):
        h=F.relu(self.fc4(z))
        h=F.relu(self.fc5(h))
        
        return self.fc6(h)
    
    def forward(self,x):
        mu,log_var=self.encoder(x)
        z=self.sampling(mu,log_var)
        
        return self.decoder(z),mu,log_var

def loss_function(recon_x,x,mu_log_var):
    criterion=nn.MSELoss(reduction='sum')
    recon_loss=criterion(recon_x,x)

    latent_loss=-0.5*torch.sum(1+log_var-mu**2-torch.exp(log_var))
    
    return recon_loss+latent_loss,recon_loss,latent_loss

#DON'T FORGET THE OBJECT OF AE, ae, and optimizer
def train_and_get(ae,optimizer,data,epochs,coming_cycle):
    #regularize the data
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    data=(data-mean)/std
    
    n=len(data)
    index=list(range(n))
    batch_size=500
    num_batch=n//500
    
    data=torch.from_numpy(data)
    train_loss_list=[]
    train_recon_loss_list=[]
    train_latent_loss_list=[]
    
    for epoch in range(epochs):
        random.shuffle(index)
        
        running_loss=0.0
        running_recon_loss=0.0
        running_latent_loss=0.0


        for i in range(num_batch):
            inputs=data[index[i*batch_size:(i+1)*batch_size]]
            optimizer.zero_grad()
            
            recon_x,mu,log_var=vae(inputs.float().to(gpu)) 
            loss,recon_loss,latent_loss=loss_function(recon_x,inputs.float().to(gpu))
            loss.backward()
            
            running_loss+=loss
            running_recon_loss+=recon_loss
            running_latent_loss+=latent_loss

            optimizer.step()
        
        train_loss_list.append(running_loss.cpu().item())
        train_recon_loss_list.append(running_recon_loss.cpu().item())
        train_latent_loss_list.append(running_latent_loss.cpu().item())

    
    plt.plot(train_loss_list)
    plt.plot(train_recon_loss_list)
    plt.plot(train_latent_loss_list)

    np.savetxt('./'+str(coming_cycle)+'loss.txt',train_loss_list)
    np.savetxt('./'+str(coming_cycle)+'recon_loss.txt',train_recon_loss_list)
    np.savetxt('./'+str(coming_cycle)+'latent_loss.txt',train_latent_loss_list)

    plt.savefig('./'+str(coming_cycle)+'/loss.png',dpi=400)
    
    PATH='./'+str(coming_cycle)+'/AE.pth'
    torch.save(vae.cpu().state_dict(),PATH)
    
    with torch.no_grad():
        recon_x,mu,log_var=vae(data.float().to(gpu))
        np.savetxt('./'+str(coming_cycle)+'/mu_all.txt',mu.cpu().numpy())
        np.savetxt('./'+str(coming_cycle)+'/std_all.txt',0.5*log_var.cpu().exp().numpy())
    
    
        
            
    
    
