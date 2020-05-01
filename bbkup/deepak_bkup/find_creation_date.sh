!/usr/bin/bash
#Find which device the file/folder is placed on:
device = df <filename>

#Find the inode number for this file/folder
inodenum = stat -c %i <filename>

#Using inode number now check the creation time

sudo debugfs -R 'stat <inodenum>' device
