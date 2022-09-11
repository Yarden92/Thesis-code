FOLDER_NAME="stas_config"
PATH="./Desktop/$FOLDER_NAME"
SSH_PATH="/home/yarcoh/projects/thesis-code4/config"
if [ -d PATH ]; then
    echo "Folder $PATH already exists,.. unmounting and deleting"
    umount $PATH
    rm -r $PATH
    echo "unmounted and removed! now recreating.."
else
    echo "creating shared folder under $PATH"
fi
mkdir $PATH
sshfs -o allow_other,default_permissions yarcoh@132.72.49.187:$SSH_PATH $PATH
echo "all set! now you can access the folder $PATH"