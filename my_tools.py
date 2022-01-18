import numpy as np

def drvtv_1(f,h,axis=0,order=2):

  '''
  First derivative of an array computed in a finite difference approximation 
  to various orders. Also the boundaries are computed to the given order.
  '''
  
  # make the axis to be operated on the first axis
  f = np.swapaxes(f,0,axis)
  
  # initialize derivativ
  df = np.zeros(np.shape(f),dtype=f.dtype)
  
  if order not in [2,4,6,8]:
    print(' drvtv_1 WARNING: Only 2nd, 4th, 6th, 8th order derivatives implemented. Falling back to 2nd order.')
    order=2
    
  # first derivative up to second order
  if order == 2:
    for ii in range(len(f)):
      if ii == 0:
        df[ii] = -3*f[ii]+4*f[ii+1]-f[ii+2]
      elif ii == len(f)-1:
        df[ii] =  3*f[ii]-4*f[ii-1]+f[ii-2]
      else:
        df[ii] = -f[ii-1]+f[ii+1]
    df /= float(2*h)
  # first derivative up to fourth order
  elif order == 4:
    for ii in range(len(f)):
      if ii == 0:
        df[ii] = -25*f[ii]+48*f[ii+1]-36*f[ii+2]+16*f[ii+3]-3*f[ii+4]
      elif ii == len(f)-1:
        df[ii] =  25*f[ii]-48*f[ii-1]+36*f[ii-2]-16*f[ii-3]+3*f[ii-4]
      elif ii == 1:
        df[ii] = -3*f[ii-1]-10*f[ii]+18*f[ii+1]-6*f[ii+2]+f[ii+3]
      elif ii == len(f)-2:
        df[ii] =  3*f[ii+1]+10*f[ii]-18*f[ii-1]+6*f[ii-2]-f[ii-3]
      else:
        df[ii] = f[ii-2]-8*f[ii-1]+8*f[ii+1]-f[ii+2]
    df /= float(12*h)
  # first derivative up to sixth order
  elif order == 6:
    for ii in range(len(f)):
      if ii == 0:
        df[ii] = -137*f[ii]+300*f[ii+1]-300*f[ii+2]+200*f[ii+3]-75*f[ii+4]+12*f[ii+5]
      elif ii == len(f)-1:
        df[ii] =  137*f[ii]-300*f[ii-1]+300*f[ii-2]-200*f[ii-3]+75*f[ii-4]-12*f[ii-5]
      elif ii == 1:
        df[ii] = -12*f[ii-1]-65*f[ii]+120*f[ii+1]-60*f[ii+2]+20*f[ii+3]-3*f[ii+4]
      elif ii == len(f)-2:
        df[ii] =  12*f[ii+1]+65*f[ii]-120*f[ii-1]+60*f[ii-2]-20*f[ii-3]+3*f[ii-4]
      elif ii == 2:
        df[ii] =  3*f[ii-2]-30*f[ii-1]-20*f[ii]+60*f[ii+1]-15*f[ii+2]+2*f[ii+3]
      elif ii == len(f)-3:
        df[ii] = -3*f[ii+2]+30*f[ii+1]+20*f[ii]-60*f[ii-1]+15*f[ii-2]-2*f[ii-3]
      else:
        df[ii] = -f[ii-3]+9*f[ii-2]-45*f[ii-1]\
                           +45*f[ii+1]-9*f[ii+2]+f[ii+3]
    df /= float(60*h)
  # first derivative up to eight order
  elif order == 8:
    for ii in range(len(f)):
      if ii == 0:
        df[ii] = 14*(-137*f[ii]+300*f[ii+1]-300*f[ii+2]+200*f[ii+3]-75*f[ii+4]+12*f[ii+5])
      elif ii == len(f)-1:
        df[ii] = 14*( 137*f[ii]-300*f[ii-1]+300*f[ii-2]-200*f[ii-3]+75*f[ii-4]-12*f[ii-5])
      elif ii == 1:
        df[ii] = 14*(-12*f[ii-1]-65*f[ii]+120*f[ii+1]-60*f[ii+2]+20*f[ii+3]-3*f[ii+4])
      elif ii == len(f)-2:
        df[ii] = 14*( 12*f[ii+1]+65*f[ii]-120*f[ii-1]+60*f[ii-2]-20*f[ii-3]+3*f[ii-4])
      elif ii == 2:
        df[ii] = 14*( 3*f[ii-2]-30*f[ii-1]-20*f[ii]+60*f[ii+1]-15*f[ii+2]+2*f[ii+3])
      elif ii == len(f)-3:
        df[ii] = 14*(-3*f[ii+2]+30*f[ii+1]+20*f[ii]-60*f[ii-1]+15*f[ii-2]-2*f[ii-3])
      elif ii == 3:
        df[ii] = 14*(-12*f[ii-1]-65*f[ii]+120*f[ii+1]-60*f[ii+2]+20*f[ii+3]-3*f[ii+4])
      elif ii == len(f)-4:
        df[ii] = 14*( 12*f[ii+1]+65*f[ii]-120*f[ii-1]+60*f[ii-2]-20*f[ii-3]+3*f[ii-4])
      else:
        df[ii] = \
            3*f[ii-4]-32*f[ii-3]+168*f[ii-2]-672*f[ii-1]\
           +672*f[ii+1]-168*f[ii+2]+32*f[ii+3]-3*f[ii+4]
    df /= (840*h)
  
  # return in correct order
  return np.swapaxes(df,0,axis)
