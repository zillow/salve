

export outf=2022_09_29_inference
mkdir -p ${outf}

TEST_BUILDING_IDS=(
	0963
	1169
	0203
	1041
	0957
	0496
	1544
	0668
	0299
	0684
	0534
	1348
	0127
	0494
	0519
	0854
	1184
	1566
	0629
	1409
	1485
	1001
	1177
	1388
	0245
	1400
	0576
	0100
	0190
	1488
	1330
	0038
	1494
	0189
	0569
	0964
	1069
	0109
	0431
	0778
	0660
	1500
	0880
	1203
	0709
	0308
	0141
	0663
	1218
	0336
	0028
	0338
	1239
	1130
	1328
	0385
	1160
	1248
	0075
	1185
	1207
	1175
	0311
	0675
	1401
	0528
	0670
	0322
	1167
	0453
	0490
	1436
	1490
	0115
	0420
	0506
	0354
	1214
	0541
	0583
	0870
	0297
	0325
	1506
	0152
	0429
	1068
	1025
	0941
	0406
	1028
	1404
	0744
	0691
	0057
	1027
	0785
	1268
	1050
	0353
	0575
	0502
	1398
	0944
	0792
	0383
	0090
	0375
	1153
	1103
	0642
	0278
	0097
	0986
	0850
	0011
	0218
	0181
	0039
	0270
	1075
	1383
	0969
	0819
	1538
	0302
	1489
	0010
	0076
	0905
	0715
	0564
	0800
	1199
	0165
	1210
	1479
	0444
	0438
	0681
	0316
	0579
	0588
	0605
	0742
	0635
	1326
	1317
	0516
	0157
	0382
	1368)


	# 0990
	# 0021
	# 0809
	# 0966
	# 1551
	# 1079)


for building_id in ${TEST_BUILDING_IDS[@]}; do

	echo " "
	echo "Evaluate on: ${building_id}"

	#sbatch --dependency=singleton --job-name=mseg_eval_A -c 5 -p short -x jarvis,vicki,cortana,gideon,ephemeral-3 --gres=gpu:1 \
	sbatch -c 5 --job-name=salve_inference_overcap_A --gres=gpu:1 -p short \
		-o ${outf}/${building_id}.log salve_inference_1gpu.sh ${building_id}
	    #--exclude crushinator,clank,rosie,siri,droid,olivaw,glados,xaea-12,nestor,heistotron,cyborg,dave,robby,megabot,sonny,spd-13,randotron,sophon,chomps,alexa,chappie,bmo,fiona,cortana,irona,ephemeral-3,vincent \
	    

	echo " "

done