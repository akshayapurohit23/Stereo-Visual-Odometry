#include <stdio.h>
#include <iostream>
#include <string.h>
#include <fstream>

#include "system.h"

int main(int argc, const char* argv[]){
	if(argc != 3){
		cout << "wrong input! " << endl;
		exit;
	}

	SLAMsystem((string)argv[1], (string)argv[2]);
	return 0;
}