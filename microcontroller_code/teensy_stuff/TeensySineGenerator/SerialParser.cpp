
#include "SerialParser.h"

//int SerialParser::MAX_ARGS = 10

SerialParser::SerialParser(interpreter interp) {
    _interp = interp;
    _usbMessage = "";
}

SerialParser::~SerialParser() {}

void SerialParser::update() {
    // 2. Read from USB, if available
    if (Serial.available() > 0) {
        // read next char if available
        char inByte = Serial.read();
        if ((inByte == '\n') || (inByte == ';')) {
            // the new-line character ('\n') or ';'
            // indicate a complete message
            // so interprete the message and then clear buffer
            parse();
            _usbMessage = ""; // clear message buffer
        } else {
            // append character to message buffer
            _usbMessage = _usbMessage + inByte;
        }
    }
}


void SerialParser::parse() {
    _usbMessage.trim(); // remove leading and trailing white space
    int len = _usbMessage.length();
    if (len==0) {
        Serial.print("#"); // "#" means error
        return;
    }

    // extract command string
    int cmdLength = 0;
    // try: (strchr("_!@#$*", _usbMessage[cmdLength]) == NULL) // need #include <string.h>
    while (isAlpha(_usbMessage[cmdLength])
            || isPunct(_usbMessage[cmdLength]) ) {
            // || (strchr("_!@#$*", _usbMessage[cmdLength]) != NULL) ) {
            // || (_usbMessage[cmdLength] == '_')
            // || (_usbMessage[cmdLength] == '$') ) {
        _cmd[cmdLength] = _usbMessage[cmdLength];
        cmdLength++;
    }
    _cmd[cmdLength] = '\0';

    // extract args
    String argString = _usbMessage.substring(cmdLength);
    int numArgs = 0;
    for (int argNum=0; argNum<MAX_ARGS; argNum++) {
        String intString = "";
        argString.trim();
        if ((argString.length() > 0) && (argString[0] == '-')) {
            intString += '-';
            argString.remove(0,1);
        }
        if ((argString.length() == 0) || (!isDigit(argString[0]))) {
            break;
        } else {
            numArgs++;
            while ((argString.length() > 0) && (isDigit(argString[0]))) {
                intString += argString[0];
                argString.remove(0,1);
            }
            _args[numArgs-1] = intString.toInt();
        }
    }
    _interp(_cmd, numArgs, _args);
}
