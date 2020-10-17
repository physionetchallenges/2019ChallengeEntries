#!/usr/bin/perl
#
@all=glob "tmp_inputs/*";

system "rm -rf training_processed";
system "mkdir training_processed";

foreach $file (@all){
	@ref=();
	@gold=();
	open FILE, "$file" or die;
	<FILE>;
	$i=0;
	while ($line=<FILE>){
		chomp $line;
		@table=split '\|', $line;
		$gold[$i]=$table[-1];
		pop @table;
		$j=0;
		foreach $aaa (@table){
			$ref[$i][$j]=$aaa;
			$j++;
		}

		$i++;
	}
	$imax=$i;
	$jmax=$j;
	close FILE;
	@t=split '/',$file;
	open NEW, ">training_processed/$t[-1]" or die;
	$i=0;
	while ($i<$imax){
		$j=0;
		if ($ref[$i][$j] eq "NaN"){
			print NEW "-3000";
		}else{
			print NEW "$ref[$i][$j]";
		}
		if ($ref[$i][$j] eq "NaN"){
			print NEW "\t0";
		}else{
			print NEW "\t1";
		}
		$j++;
		while ($j<$jmax){
			if ($ref[$i][$j] eq "NaN"){
				print NEW "\t-3000";
			}else{
				print NEW "\t$ref[$i][$j]";
			}
			if ($ref[$i][$j] eq "NaN"){
				print NEW "\t0";
			}else{
				print NEW "\t1";
			}
			$j++;
		}
		print NEW "\n";

		$i++;
	}
	close NEW;


	#open NEW, ">training_processed/$t[-1].gs" or die;
	#$i=0;
	#while ($i<$imax){
	#	print NEW "$gold[$i]\n";
	#	$i++;
	#}
	#close NEW;
	
}
