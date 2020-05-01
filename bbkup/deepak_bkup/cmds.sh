#!/bin/sh
rm /var/lib/apt/lists/lock
rm /var/cache/apt/archives/lock
apt-get update --fix-missing
apt-get install -f
apt-get upgrade -f
apt-get install --reinstall gnome-session
apt-get install --reinstall ubuntu-desktop
