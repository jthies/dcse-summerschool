#!/bin/bash

mkdir -p cd /scratch/$USER/suite-sparse-matrices
cd suite-sparse-matrices

matrices="af_shell10 kkt_power Transport"
webLinks="https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz https://suitesparse-collection-website.herokuapp.com/MM/Zaoui/kkt_power.tar.gz https://suitesparse-collection-website.herokuapp.com/MM/Janna/Transport.tar.gz"

ctr=1
for matrix in $matrices; do
	webLink=$(echo ${webLinks} | cut -d" " -f${ctr})
	if [ ! -f "${matrix}.mtx" ]; then
		echo "Downloading ${matrix} matrix from SuiteSparse collection"
		wget ${webLink}

		echo "Unpacking ${matrix} matrix"
		tar -xvzf ${matrix}.tar.gz
		mv ${matrix}/${matrix}.mtx .

		#remove tars and folders
		rm -rf ${matrix} ${matrix}.tar.gz
	else
		echo "Found ${matrix} matrix"
	fi
	let ctr=${ctr}+1
done

cd -
echo "Done"
