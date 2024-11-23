import os

cmd = ''
flags = ''

while not os.path.exists(flags):
  print(cmd)
  os.system(cmd)
