echo """
run the following commands (dont copy):
  ssh yarcoh@132.72.49.187 du -hd1 ./projects/thesis-code4/data/datasets
  scp -r yarcoh@132.72.49.187:projects/thesis-code4/data/... ~/Desktop/projects/thesis_stuff/Thesis-code/data/...
"""
echo "here are the datasets:"
ssh yarcoh@132.72.49.187 du -hd2 ./projects/thesis-code4/data/datasets

echo "\nand here are the models:"
ssh yarcoh@132.72.49.187 du -hd1 ./projects/thesis-code4/data/saved_models

echo ""
read -p "enter the folder path after ../data/[..HERE..]: " name
echo "

copying
  from: <ssh>/projects/thesis-code4/data/$name
  to: <local>/Desktop/projects/thesis_stuff/Thesis-code/data/$name

"
while true; do
    read -p "Confirm? (Y/N) " yn
    case $yn in
        [Yy]* ) scp -r yarcoh@132.72.49.187:projects/thesis-code4/data/$name ~/Desktop/projects/thesis_stuff/Thesis-code/data/$name; break;;
        [Nn]* ) echo "np, canceling..";exit;;
        * ) echo "Please answer Y or N.";;
    esac
done
