#ifndef _SerialParser_h_
#define _SerialParser_h_

#include <Arduino.h>

typedef void (*interpreter)(char* cmd, int num_args, long* args);

class SerialParser
{
public:
	static const int MAX_ARGS = 10;
	static const int MAX_CMD = 128;
private:
	HardwareSerial _serial;
	interpreter _interp;
	String _usbMessage;
	long _args[MAX_ARGS];
	char _cmd[MAX_CMD+1];
public:
	SerialParser(interpreter interp);
	~SerialParser();
	void update();
	void parse();
};


#endif	//_SerialParser_h_
