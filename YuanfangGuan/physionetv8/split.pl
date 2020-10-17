#!/usr/bin/perl
#
#
srand($ARGV[0]);

@all_example=glob "../../rawdata_updated/all/*";

system "rm -rf training_data";
system "mkdir training_data";


open TRAIN, ">train_gs.dat" or die;
open TEST, ">test_gs.dat" or die;
foreach $example (@all_example){
	$r=rand(1);
    $ori_example=$example;
	$example=~s/all/all_processed/;
	$gs=$example.'.gs';
	if ($r<0.999){
		print TRAIN "$ori_example\t$gs\n";
	}else{
            system "cp $ori_example training_data/";
		print TEST "$example\t$gs\n";
	}
}
close TRAIN;
close TEST;


