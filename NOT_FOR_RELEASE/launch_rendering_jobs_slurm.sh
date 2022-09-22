#!/bin/bash


export outf=2022_07_18_inference
mkdir -p ${outf}


BUILDING_IDS=(
	"0691"
	"1001"
	"0579"
	"0299"
	"0245"
	"0963"
	"1130"
	"1490"
	"1160"
	"0420"
	"0429"
	"1388"
	"1398"
	"1544"
	"1184"
	"1068"
	"1538"
	"0605"
	"0870"
	"0528"
	"1404"
	"1328"
	"0681"
	"0308"
	"1050"
	"0534"
	"1207"
	"1185"
	"1368"
	"0629"
	"0496"
	"0453"
	"1383"
	"0957"
	"0792"
	"1041"
	"0905"
	"0564"
	"1203"
	"0969"
	"1479"
	"1494"
	"0431"
	"1401"
	"0354"
	"1551"
	"1075"
	"1027"
	"0406"
	"0353"
	"1239"
	"1218"
	"1566"
	"1069"
	"1210"
	"1330"
	"0490"
	"1500"
	"0583"
	"1248"
	)


for building_id in ${BUILDING_IDS[@]}; do

	echo " "
	echo "Evaluate on: ${building_id}"

	#sbatch --dependency=singleton --job-name=mseg_eval_A -c 5 -p short -x jarvis,vicki,cortana,gideon,ephemeral-3 --gres=gpu:1 \
	sbatch --gres gpu:1 -c 5 --constraint=a40 --job-name=salve_rendering_overcap_A \
	    #--exclude crushinator,clank,rosie,siri,droid,olivaw,glados,xaea-12,nestor,heistotron,cyborg,dave,robby,megabot,sonny,spd-13,randotron,sophon,chomps,alexa,chappie,bmo,fiona,cortana,irona,ephemeral-3,vincent \
	    -o ${outf}/${building_id}.log render_single_zind_building.sh ${building_id}

	echo " "

done