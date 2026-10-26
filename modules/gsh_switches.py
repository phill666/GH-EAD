"""Optional GSHConv switches, isolated from the project's default model factory."""
from __future__ import annotations
import torch
from torch import nn
from modules.gs_halfconv_v2 import GSHalfConvV2
class ConfigurableGSHalfConvV2(GSHalfConvV2):
 def __init__(self,channels:int,groups:int=2,half_ratio:float=0.5,shuffle:bool=True,se_gate:bool=True,**kwargs):
  if groups not in (1,2,3,4):raise ValueError('Supported sensitivity groups: 1,2,3,4')
  if half_ratio not in (0.25,0.5,0.75):raise ValueError('Supported half ratios: .25,.50,.75')
  super().__init__(channels,groups=groups,**kwargs)
  self.revision_half_ratio=float(half_ratio);self.revision_shuffle=bool(shuffle);self.revision_se_gate=bool(se_gate)
  if half_ratio!=0.5:
   for size,block in zip(self.group_channels,self.partial_blocks):
    count=max(1,min(size,int(size*half_ratio)))
    block.conv_channels=count;block.skip_channels=size-count
    block.partial_conv=nn.Conv2d(count,count,kernel_size=3,stride=1,padding=1,bias=False)
  if not shuffle:self.shuffle_groups=1
  if not se_gate:self.channel_gate=nn.Identity()
  self.revision_uneven_shuffle=shuffle and groups>1 and channels%groups!=0
  if self.revision_uneven_shuffle:
   offsets=[];offset=0
   for size in self.group_channels:offsets.append(offset);offset+=size
   permutation=[offsets[g]+i for i in range(max(self.group_channels)) for g,size in enumerate(self.group_channels) if i<size]
   assert sorted(permutation)==list(range(channels))
   self.register_buffer('revision_shuffle_index',torch.tensor(permutation,dtype=torch.long),persistent=False)
 def forward(self,x):
  if not self.revision_uneven_shuffle:return super().forward(x)
  # Extension only for unequal groups: same channel interleaving as the original shuffle.
  pieces=torch.split(x,self.group_channels,dim=1)
  result=torch.cat([block(part) for block,part in zip(self.partial_blocks,pieces)],dim=1)
  result=result.index_select(1,self.revision_shuffle_index)
  result=self.norm(self.channel_gate(self.mixer(result)))
  return x+self.res_scale*result

def install_gsh_switches(settings):
 """Patch only the in-memory factory used by a dedicated new revision run."""
 import efficientad_variant_gs_hf_v2 as factory
 from functools import partial
 allowed={'groups','half_ratio','shuffle','se_gate'}
 if set(settings)-allowed:raise ValueError('Unknown GSH settings: '+str(set(settings)-allowed))
 values={'groups':2,'half_ratio':0.5,'shuffle':True,'se_gate':True,**settings}
 factory.GSHalfConvV2=partial(ConfigurableGSHalfConvV2,**values)
 return values
