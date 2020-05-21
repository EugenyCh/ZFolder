#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <map>

using namespace std;

int main()
{
	ifstream in("input.bin", ios::binary);
	if (!in)
	{
		cout << "Error of opening \"input.bin\"" << endl;
		return 1;
	}

	string initString;
	int rulesCount;
	map<char, string> rules;
	int startIter;
	float width0, width1;
	unsigned char color0[4], color1[4];
	float angle, scaling;
	int temp;
	char* strTemp;

	// Rules count
	in.read((char*)&rulesCount, sizeof(int));

	// Init string length
	in.read((char*)&temp, sizeof(int));

	// Init string
	strTemp = new char[temp + 1];
	in.read((char*)strTemp, temp);
	strTemp[temp] = 0;
	initString = strTemp;
	delete[] strTemp;

	for (int i = 0; i < rulesCount; ++i)
	{
		// Rule letter
		char ruleName;
		in.read((char*)&ruleName, 1);
		// Rule string length
		in.read((char*)&temp, sizeof(int));
		// Relu definition
		strTemp = new char[temp + 1];
		in.read((char*)strTemp, temp);
		strTemp[temp] = 0;
		rules[ruleName] = strTemp;
		delete[] strTemp;
	}

	// Start iteration
	in.read((char*)&startIter, sizeof(int));

	// Color 0
	in.read((char*)&color0, 4);

	// Color 1
	in.read((char*)&color1, 4);

	// Width 0
	in.read((char*)&width0, sizeof(float));

	// Width 1
	in.read((char*)&width1, sizeof(float));

	// Angle
	in.read((char*)&angle, sizeof(float));

	// Scaling
	in.read((char*)&scaling, sizeof(float));

	in.close();

	cout << "Rules count : " << rulesCount << endl;
	cout << "Init string (" << initString.length() << " bytes) : " << initString << endl;
	for (auto r : rules)
	{
		cout << "Rule '" << r.first << "' (" << r.second.length() << " bytes) : " << r.second << endl;
	}
	cout << "Start iteration : " << startIter << endl;
	cout << "Color 0 : ["
		<< (int)(color0[0]) << ", "
		<< (int)(color0[1]) << ", "
		<< (int)(color0[2]) << ", "
		<< (int)(color0[3]) << "]" << endl;
	cout << "Color 1 : ["
		<< (int)(color1[0]) << ", "
		<< (int)(color1[1]) << ", "
		<< (int)(color1[2]) << ", "
		<< (int)(color1[3]) << "]" << endl;
	cout << "Width 0 : " << width0 << endl;
	cout << "Width 1 : " << width1 << endl;
	cout << "Angle : " << angle << " (" << (angle * 180.0 / M_PI) << " degrees)" << endl;
	cout << "Scaling : " << scaling << endl;

	return 0;
}
