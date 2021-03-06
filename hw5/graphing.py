import matplotlib.pyplot as plt
from pylab import *
import math


plt.clf()
xs = range(1,15)
ys2 = [15.4,16.8666666667,20.6666666667,18.6666666667,16.1333333333,15.6,17.0666666667,16.6,17.1333333333,17.6,16.4666666667,17.9333333333,14.8666666667,19.9333333333]
ys = [13.0666666667,16.7333333333,16.8666666667,19.1333333333,17.3333333333,16.2666666667,15.4666666667,16.5333333333,15.2666666667,16.2666666667,14.6,14.2666666667,15.2,16.8]
p1, = plt.plot(xs, ys, color='b')
p2, = plt.plot(xs, ys2, color='r')
plt.title('Performance vs Epoch Size')
plt.xlabel('Epoch Size')
plt.ylabel('Average number of throws')
plt.legend((p1,p2), ('strategy 1','strategy 2'), 'upper right')
savefig('modelbased.jpg') # save the figure to a file
# plt.show() # show the figure

plt.clf()
xs = [1,5,10,15]
ys = [16.14,16.86,31.65,19.31]
ys2 = [16.29,14.53,16.94,16.49]
p1, = plt.plot(xs, ys, color='b')
p2, = plt.plot(xs, ys2, color='r')
plt.title('Performance vs Epoch Size')
plt.xlabel('Epoch Size')
plt.ylabel('Average number of throws')
plt.legend((p1,p2), ('strategy 1','strategy 2'), 'upper right')
savefig('modelbased100.jpg') # save the figure to a file
# plt.show() # show the figure

#stupid way

# EPOCH SIZE:  1
# Thrower Bias: -0.30883152611 -0.281249113252
# Thrower wobble: 0.561347256903 0.501712370625
# Average turns =  15.4
# EPOCH SIZE:  2
# Thrower Bias: -0.0801365376305 0.0986784136061
# Thrower wobble: 0.739094010428 0.601906701559
# Average turns =  16.8666666667
# EPOCH SIZE:  3
# Thrower Bias: -0.278383309494 0.62115945304
# Thrower wobble: 0.661199163674 0.7054018279
# Average turns =  20.6666666667
# EPOCH SIZE:  4
# Thrower Bias: 0.644385498049 -0.209055449645
# Thrower wobble: 0.538125970878 0.627643693281
# Average turns =  18.6666666667
# EPOCH SIZE:  5
# Thrower Bias: -0.399519370476 -0.105490654568
# Thrower wobble: 0.545608126094 0.660501763722
# Average turns =  16.1333333333
# EPOCH SIZE:  6
# Thrower Bias: -0.457270053487 -0.0895210100851
# Thrower wobble: 0.639456404094 0.525452167397
# Average turns =  15.6
# EPOCH SIZE:  7
# Thrower Bias: 0.0519937563189 0.345443582146
# Thrower wobble: 0.708509088444 0.750910325807
# Average turns =  17.0666666667
# EPOCH SIZE:  8
# Thrower Bias: -0.0223355891359 0.169626022485
# Thrower wobble: 0.502903527759 0.759550933038
# Average turns =  16.6
# EPOCH SIZE:  9
# Thrower Bias: 0.260002405008 0.543762917964
# Thrower wobble: 0.546417439026 0.635273471265
# Average turns =  17.1333333333
# EPOCH SIZE:  10
# Thrower Bias: 0.369324992565 -0.419869038211
# Thrower wobble: 0.606763599915 0.76265736496
# Average turns =  17.6
# EPOCH SIZE:  11
# Thrower Bias: -0.0333129718463 0.196564145382
# Thrower wobble: 0.615205020955 0.666892414098
# Average turns =  16.4666666667
# EPOCH SIZE:  12
# Thrower Bias: 0.0149002809019 0.33800624454
# Thrower wobble: 0.557391123704 0.78945286862
# Average turns =  17.9333333333
# EPOCH SIZE:  13
# Thrower Bias: -0.139003790126 -0.305549468664
# Thrower wobble: 0.540627678629 0.686724289393
# Average turns =  14.8666666667
# EPOCH SIZE:  14
# Thrower Bias: -0.116918554604 0.626740971815
# Thrower wobble: 0.691966984241 0.748816041426
# Average turns =  19.9333333333


# #epsilon

# EPOCH SIZE:  1
# Thrower Bias: 0.257855703187 -0.409203373737
# Thrower wobble: 0.565769031294 0.578239300245
# Average turns =  13.0666666667
# EPOCH SIZE:  2
# Thrower Bias: -0.0735791184455 -0.373479776954
# Thrower wobble: 0.647383669959 0.624038739619
# Average turns =  16.7333333333
# EPOCH SIZE:  3
# Thrower Bias: -0.327006481842 0.356950234233
# Thrower wobble: 0.638334624385 0.544863545882
# Average turns =  16.8666666667
# EPOCH SIZE:  4
# Thrower Bias: 0.163459822548 0.483880615641
# Thrower wobble: 0.767999888218 0.529725366833
# Average turns =  19.1333333333
# EPOCH SIZE:  5
# Thrower Bias: 0.0477887922363 -0.611459014207
# Thrower wobble: 0.517539827477 0.61205739648
# Average turns =  17.3333333333
# EPOCH SIZE:  6
# Thrower Bias: 0.356830530539 0.0366850534871
# Thrower wobble: 0.689665053935 0.660145258248
# Average turns =  16.2666666667
# EPOCH SIZE:  7
# Thrower Bias: 0.258956403749 0.443038732478
# Thrower wobble: 0.504312339265 0.668259934735
# Average turns =  15.4666666667
# EPOCH SIZE:  8
# Thrower Bias: -0.326277314354 -0.0961771468309
# Thrower wobble: 0.718316895077 0.656457389991
# Average turns =  16.5333333333
# EPOCH SIZE:  9
# Thrower Bias: -0.0758438686196 -0.00898830986014
# Thrower wobble: 0.745145204174 0.605833826948
# Average turns =  15.2666666667
# EPOCH SIZE:  10
# Thrower Bias: -0.00058991695026 0.369927029796
# Thrower wobble: 0.614170964773 0.694089006585
# Average turns =  16.2666666667
# EPOCH SIZE:  11
# Thrower Bias: 0.194833690471 0.049056770447
# Thrower wobble: 0.521998369553 0.577917097389
# Average turns =  14.6
# EPOCH SIZE:  12
# Thrower Bias: 0.0889793813725 0.140870438122
# Thrower wobble: 0.533610468536 0.516050782236
# Average turns =  14.2666666667
# EPOCH SIZE:  13
# Thrower Bias: -0.271639990452 0.396790367012
# Thrower wobble: 0.606070007768 0.522025648767
# Average turns =  15.2
# EPOCH SIZE:  14
# Thrower Bias: -0.127370690393 -0.0590328911341
# Thrower wobble: 0.785130552504 0.559204450794
# Average turns =  16.8

