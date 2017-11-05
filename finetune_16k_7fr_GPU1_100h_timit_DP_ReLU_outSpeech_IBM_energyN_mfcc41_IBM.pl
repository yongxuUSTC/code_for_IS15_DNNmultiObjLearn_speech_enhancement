use strict;

my $i;
my $j;
my $line;
my $curacc;
my $preacc;
#my $threshold=0.1;


my $numlayers=5;

	my $lrate=0.1; ####dropout的学习速率，大10倍
	my $layersizes = "2384"; # (257+41)*7+(257+41)
	for(my $i=0;$i<$numlayers -2;$i++)
	{
		$layersizes	  .= ",2500";
	}	
	$layersizes	  .= ",555";# (257+41)+257
	
	my $node=2500;
	
#	my $hidname = "";
#	for(my $i=0;$i<$numlayers -2;$i++)
#	{
#		$hidname	  .= "_h500";
#	}	

	my $exe 						= "./code_BP_GPU_for_SE_Dropout_ReLU_NAT_outputSpeech0.5And0.2IBM_GPU1_16k_energyN_2err_bMFCC41dim0.3/BPtrain";
	my $gpu_used				= 1;
#	my $numlayers				= 4;
#	my $layersizes			= "429,1024,1024,183";

	my $bunchsize				= 128;#128
	my $momentum				= 0.5;
	my $weightcost			= 0;
	my $fea_dim					= 298;#(257+41)
	my $fea_context			= 7;
	my $traincache			= 102400;  ############ how many samples per chunk #102400
	my $init_randem_seed= 27863875;   ############ every epoch must change
	my $targ_offset			= 3;
	
#	my $CF_DIR					= "config";
#	my $norm_file				= "$CF_DIR/fea_tr.norm_data_timit_SNR_20_15_10_5_0_-5";
#	my $fea_file				= "$CF_DIR/timit_Multi_NT_SNR_100h_all_trainset_25cases_random_ts2000_noisy.pfile";
#	my $targ_file				= "$CF_DIR/timit_Multi_NT_SNR_100h_all_trainset_25cases_random_ts2000_clean.pfile";########################
#	my $CF_DIR					= "/home/yongxu/step1_prepare_data/data_timit_104NT_2500h_from19/pretrain_pfile/";
#	my $norm_file				= "$CF_DIR/104NT_7SNRs_2500h_EachCase4H_trainset_random_ts2500.fea_norm";
#	my $fea_file				= "$CF_DIR/104NT_7SNRs_2500h_EachCase4H_trainset_random_ts2500_noisy_linux.pfile";
#	my $targ_file				= "$CF_DIR/104NT_7SNRs_2500h_EachCase4H_trainset_random_ts2500_clean_linux.pfile";########################
#	my $CF_DIR					= "/home/yongxu/step1_prepare_data/data_timit_104NT_2500h_from19/pretrain_pfile/get_100h_104NT_7SNRs_random_ts2500";
#	my $norm_file				= "$CF_DIR/104NT_7SNRs_2500h_EachCase4H_trainset_random_ts2500.fea_norm";
#	my $fea_file				= "/home/yongxu/step1_prepare_data/data_timit_104NT_2500h_from19/pretrain_pfile/get_100h_104NT_7SNRs_random_ts2500/104NT_7SNRs_100h_EachCase4H_trainset_random_ts2500_noisy_linux.pfile";
#	my $targ_file				= "/home/yongxu/step1_prepare_data/data_timit_104NT_100h_noisePfile_from18/pretrain_pfile/104NT_7SNRs_100h_EachCase4H_trainset_random_ts2500_clean_and_noise_linux_NEW.pfile";#########
	
#		my $CF_DIR					= "/mnt/45.94_yongxu_d2/config/get_timit_aurora4_16k_102NT_200h";
#	my $norm_file				= "$CF_DIR/timit_aurora4_102NT_7SNRs_each190_80utts_noisy_lsp_be_random_linux.fea_norm";
#	my $fea_file				= "$CF_DIR/timit_aurora4_102NT_7SNRs_each190_80utts_noisy_lsp_be_random_linux.pfile";
#	my $targ_file				= "$CF_DIR/timit_aurora4_102NT_7SNRs_each190_80utts_cleanANDnoise_lsp_be_random_linux.pfile";#########
#			my $CF_DIR					= "/mnt/45.236_gaotian/SE_100h/data_prepare";
#	my $norm_file				= "$CF_DIR/timit_aurora4_115NT_7SNRs_each190_80utts_noisy_lsp_be_random_linux.fea_norm";
#	my $fea_file				= "$CF_DIR/timit_aurora4_115NT_7SNRs_each190_80utts_noisy_lsp_be_random_linux.pfile";
#	my $targ_file				= "$CF_DIR/timit_aurora4_115NT_7SNRs_each190_80utts_clean_AND_noise_lsp_be_random_linux.pfile";#########
				#my $CF_DIR					= "/disk1/yongxu_d1/data_timit_16k_115NT_100h";
#				my $CF_DIR					= "/disk1/yongxu_d1/data_timit_16k_115NT_100h";
#	my $norm_file				= "$CF_DIR/timit_115NT_7SNRs_each190utts_noisy_lspANDmfcc_be_random_linux.fea_norm";
#	my $fea_file				= "$CF_DIR/timit_115NT_7SNRs_each190utts_noisy_lspANDmfcc_be_random_linux.pfile";
#	my $targ_file				= "$CF_DIR/timit_115NT_7SNRs_each190utts_clean_lspANDmfcc_noiseLSP_be_random_linux.pfile";#########
				my $CF_DIR					= "/disk4/yongxu_d4/get_timit_115NT_16K_pfile";
	my $norm_file				= "$CF_DIR/timit_115NT_7SNRs_each190utts_noisy_lspANDmfcc41FB40_be_random_linux.fea_norm";
	my $fea_file				= "$CF_DIR/timit_115NT_7SNRs_each190utts_noisy_lspANDmfcc41FB40_be_random_linux.pfile";
	my $targ_file				= "$CF_DIR/timit_115NT_7SNRs_each190utts_clean_lspANDmfcc41FB40_IBM257_be_random_linux.pfile";#########
		
#	my $train_sent_range		= "0-115499";
#	#my $train_sent_range		= "8-9";
#	my $cv_sent_range				= "115500-117499";
#	#my $cv_sent_range				= "1-1";
#	my $train_sent_range		= "0-721874"; #625h
#	#my $train_sent_range		= "8-9";
#	my $cv_sent_range				= "2887500-2889999";
#	#my $cv_sent_range				= "1-1";
	#my $train_sent_range		= "0-2887499";#2500h
#	my $train_sent_range		= "0-1443749"; #1250h
#	#my $train_sent_range		= "8-9";
#	my $cv_sent_range				= "2887500-2889999";
#	#my $cv_sent_range				= "1-1";
#	my $train_sent_range		= "0-125070";#截取200H，共165720句，约265H
#	#my $train_sent_range		= "0-6253";#截取200H，共165720句，约265H
#	my $cv_sent_range				= "164720-165719";#截取最后面的1000句作为CV集
	my $train_sent_range		= "0-96621";#截取200H，共165720句，约265H
	#my $train_sent_range		= "0-1276";#截取200H，共165720句，约265H
	#my $train_sent_range		= "0-965";#截取200H，共165720句，约265H
	my $cv_sent_range				= "96622-96922";#截取最后面的1000句作为CV集
	
	my $MLP_DIR					= "models/timit_115NT_80H_7SNRs_random_batch$bunchsize\_momentum$momentum\_frContext$fea_context\_lrate$lrate\_node$node\_numlayer$numlayers\-Rand_2384_3hid2500_555.belta0.5-F6NAT-ReLU-outS0.5And0.2IBM-DpV0.1H0.1-GPU1-energyN-2err-bMFCC0.3-mfcc41fb40-IBM0.2";###########################################################################
	
	system("mkdir $MLP_DIR");
	my $outwts_file			= "$MLP_DIR/mlp.1.wts";
	my $log_file				= "$MLP_DIR/mlp.1.log";
	my $initwts_file		= "pretraining_weights/Rand_2384_3hid2500_555.belta0.5.wts";#########################
	###my $initwts_file		= "/home/jiapan/new_BP_Code/BPtrain_v1_mlp/mlp.6.wts.right";
	
#	#printf("2");
	print "iter 1 lrate is $lrate\n"; 
	system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
	  " dropoutflag=1".
		" visible_omit=0.1".
		" hid_omit=0.1"
		);
#		
##	die;
##	
##		my $success=open LOG, "$log_file";
##		if(!$success)
##		{
##			printf "open log fail\n";
##		}
##		while(<LOG>)
##	  {
##	  	chomp;
##	  	if(/CV over.*/)
##	  	{
##	  	  s/CV over\. right num: \d+, ACC: //; 
##	  	  s/%//; 
##	  	  $curacc=$_;
##	  	}	  	
##	  }
##	  close LOG;
##	  
#  $preacc=$curacc;
#	my $destep=0;
#	########################################
##	$init_randem_seed=27865600;
##	$momentum=0.7;
	########################################
	for($i= 2;$i <= 10;$i++){

		$j = $i -1;
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		$init_randem_seed  += 345;
		$momentum=$momentum+0.04;
    #
    print "iter $i lrate is $lrate\n"; 
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=1".
		" visible_omit=0.1".
		" hid_omit=0.1"
		);
	}
		
#		my $success=open LOG, "$log_file";
#		if(!$success)
#		{
#			printf "open log fail\n";
#		}
#		while(<LOG>)
#	  {
#	  	chomp;
#	  	if(/CV over.*/)
#	  	{
#	  	  s/CV over\. right num: \d+, ACC: //; 
#	  	  s/%//; 
#	  	  $curacc=$_;
#	  	}	  	
#	  }
#	  close LOG;
#
#	  if($curacc<$preacc+$threshold)	
#	  {
#	  	print "iter $i ACC $curacc < iter $j ACC $preacc+threshold($threshold)\n";
#	  	$destep++;
#	  	print "destep is $destep\n";
#	  	if($destep>=3)
#	  	{
#	  		
#	  		unlink($outwts_file) or warn "can not delete weights file";
#	  		unlink($log_file) or warn "can not delete log file";
#	  		$i+100;
#	  		#print "finetune end\n";
#	  		last;
#	  	}
#	  	else
#	  	{
#	  	$i--;	  	
#	  	$lrate *=0.5;
# 	    }
#	  }
#	  else
#	  {
#	  	$destep=0;
#	  	$preacc=$curacc;
#	  	print "1\n\n\n\n\n\n\n\n";
#	  }
#
#	}
#	
#	########################################
#	$init_randem_seed=27878365;
#	$momentum=0.9;
#	$lrate=0.1;
#	########################################
	for($i= 11;$i <= 50;$i++){
		$j = $i -1;
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		#$lrate *= 0.9;
		$momentum=0.9;
		$init_randem_seed  += 345;
		print "iter $i lrate is $lrate\n"; 
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=1".
		" visible_omit=0.1".
		" hid_omit=0.1"
		);
	}