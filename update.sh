rsync -e ssh -rtuv ~/synthcity/workspace/* bv292.gpu2@10.147.17.202:~/uncertainty/workspace
rsync -e ssh -rtuv bv292.gpu2@10.147.17.202:~/uncertainty/workspace/* ~/synthcity/workspace

rsync -e ssh -rtuv ~/synthcity/synthetic_data/* bv292.gpu2@10.147.17.202:~/uncertainty/synthetic_data
rsync -e ssh -rtuv bv292.gpu2@10.147.17.202:~/uncertainty/synthetic_data/* ~/synthcity/synthetic_data
