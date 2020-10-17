#!/usr/bin/perl
#
system "rm -rf gs";
system "mkdir gs";
@all=glob "result/*";
foreach $file (@all){
	@t=split '/', $file;
    $t[-1]=~s/out/psv/;
	system "cp ../../rawdata_updated/all/$t[-1] ./gs/";
}
