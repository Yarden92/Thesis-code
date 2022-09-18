#! /bin/bash
mypath="${HOME}/Desktop/stas_folder"
SSH_PATH="/home/yarcoh/projects/thesis-code4"
if [ -d $mypath ]; then
    echo "Folder ${mypath} already exists,.. unmounting and deleting"
    umount $mypath
    rm -r $mypath
    echo "unmounted and removed! now recreating.."
else
    echo "creating shared folder under $mypath"
fi
mkdir $mypath
sshfs -o allow_other,default_permissions yarcoh@132.72.49.187:$SSH_PATH $mypath
echo "all set! now you can access the folder $mypath"