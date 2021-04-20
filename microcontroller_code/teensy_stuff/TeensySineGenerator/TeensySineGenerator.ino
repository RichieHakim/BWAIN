#include "SerialParser.h"
#include <string.h>
#include <math.h>


// ======
// USER MODIFIABLE DEFAULTS:
// ======

// ============
// RH EDIT, 20200314: exponential octave transform now done here instead of from input voltage. See 'RH EDIT' below.
//                    Input now assumed to scale linearly with octave spacing between minFreq and maxFreq.


// Default Sine parameters
float sinFrequency = 1000; // in Hz
int sinAmplitude = 3300; // in mV
bool freqFromPin = true; // Use analog input pin to control frequency? (T/F)
bool ampFromPin = true;  // Use analog input pin to control amplitude? (T/F)
// Set dynamic range for analog frequency control (pin input):
float minFreq = 1000;	// in Hz
float maxFreq = 18000;	// in Hz

// Hardware settings
const int sineOutputPin = A21;    // A21 is DAC0 pin on Teensy-3.5  (A14 on Teensy 3.2)
const int frequencyInputPin = A0;
const int amplitudeInputPin = A1;
const int mutePin = 1;				// set to GND to mute. Otherwise leave foating or at 3.3V
//RH added
const int voltageOutputPin = A22;
//
// ==========================




const int ANALOG_WRITE_RES = 12;
const int MAX_ANALOG_VOLTAGE_MV = 3300;

const int SIN_TABLE_SIZE = 1024;
int sinLookupTable[SIN_TABLE_SIZE];

void interpretCommand(char* command, int numArgs, long* args);
SerialParser sp = SerialParser(interpretCommand);

bool isMuted;
float amplitudeModulation = 1.0;

IntervalTimer sinTimer;
const int sinTimerInterval_us = 3;   // on Teensy 3.5: 1us is too fast; 2us works; 3us gives us some buffer

// SETUP
void setup() {
	// 1. Initialize pins
	analogWriteResolution(ANALOG_WRITE_RES);
	pinMode(sineOutputPin, OUTPUT);
	pinMode(frequencyInputPin, INPUT);
	pinMode(amplitudeInputPin, INPUT);
	pinMode(mutePin, INPUT_PULLUP);
    // RH added
    pinMode(voltageOutputPin, OUTPUT);
    // RH added
    analogWrite(voltageOutputPin, sinAmplitude);  //
	generateSinTable(sinAmplitude);

    // 2. Set up Serial
    Serial.begin(9600);

    // 3. Start analog output of sine wave
    //    This command will call updateSin() repeatedly with an interval
    //    of sinTimerInterval_us.
    sinTimer.begin(updateSin, sinTimerInterval_us);
    // sinTimer.priority(0);
}

// LOOP
void loop() {
	// The following two commands check for Serial input and update
	// the sine frequency (based on the analog input pin).
	// They will be interrupted by the analog output command, as needed.
	sp.update();
	updateParams();
}

// UTILITY COMMANDS

void generateSinTable(int amplitude_mV) {
	// convert from mV to DAC units:
	float dacAmplitude = (float)amplitude_mV * (pow(2, ANALOG_WRITE_RES)-1) / MAX_ANALOG_VOLTAGE_MV ;
	for (int i = 0; i < SIN_TABLE_SIZE; i++) {
		sinLookupTable[i] = dacAmplitude / 2 * (1 - cos(2.0*M_PI * i / SIN_TABLE_SIZE) );
	}
}

void updateParams() {
	if (freqFromPin) {
		// convert from analogRead range of (0–1023) to (minFreq–maxFreq):
		// sinFrequency = minFreq + (maxFreq - minFreq) * (long)analogRead(frequencyInputPin) / 1023L;
    // RH EDIT, 20200314: exponential octave transform now done here instead of from input voltage.
//    min_offset = (log(minFreq)/log(2));
//    max_multiplier = (log(maxFreq)/log(2));
//    input = ((float)analogRead(frequencyInputPin) / 1023.0)
//    sinFrequency = pow(2, (input*(max_multiplier-min_offset)+min_offset)); // the below equation is just an expanded version of this equation
    sinFrequency = pow(2, (((float)analogRead(frequencyInputPin) / 1023.0)*((log(maxFreq)/log(2))-(log(minFreq)/log(2)))+(log(minFreq)/log(2))));
//    Serial.println(sinFrequency);
	}
    if (digitalRead(mutePin)==LOW) {
        isMuted = true;
    } else {
        isMuted = false;
    }
    if (ampFromPin) {
        amplitudeModulation = (float)analogRead(amplitudeInputPin) / 1023.0;
//        Serial.print(amplitudeModulation);
    }
}

void updateSin() {
	// This command is called by the IntervalTimer. Keep it as fast as possible to maintain
	// a high sample generation rate. It is an interrupt callback so you cannot call
	// millis(), micros(), or any Serial commands from it.

	// This function operates like a software DDS (direct digital synthesis):
	// phaseAccumulator: Stores the current phase of the sine wave. The phase range of
	//					 0–2pi is stored as a value from 0 – (SIN_TABLE_SIZE * 1000000).
	// sinLookupTable[]: Stores the values for one full cycle of the output waveform.
	//					 Index into it using phase (0–SIN_TABLE_SIZE) as index.
	// Each time updateSin() is called, it increments the phaseAccumulator by an appropriate
	// amount (assuming that sinTimerInterval_us has passed since last call). It then uses that
	// phase to index into the lookup table to get the right analog output value.
    if (isMuted) {
        return; // do nothing
    }
	static unsigned long phaseAccumulator = 0;
	phaseAccumulator = phaseAccumulator + sinTimerInterval_us * sinFrequency * SIN_TABLE_SIZE;
	phaseAccumulator = phaseAccumulator % (SIN_TABLE_SIZE * 1000000);
    if (ampFromPin) {
        analogWrite(sineOutputPin, amplitudeModulation * sinLookupTable[round(phaseAccumulator / 1000000)]);
    } else {
        analogWrite(sineOutputPin, sinLookupTable[round(phaseAccumulator / 1000000)]);
    }
}



// Definitions of USB/Serial commands
// ==================================

void interpretCommand(char* command, int numArgs, long* args) {
	// // Uncomment this to print out commands as they are recieved:
    // Serial.print("Command: ");
    // Serial.print(command);
    // // Serial.print(" (");
    // // Serial.print(numArgs);
    // // Serial.println(" args)");
    // Serial.print(" [ ");
    // for (int i=0; i<numArgs; i++) {
    //     Serial.print(args[i]);
    //     Serial.print(" ");
    // }
    // Serial.println(" ] ");

    String msg;

    if (strcmp(command,"SetMaxFreq") == 0) {
        if (numArgs < 1) {Serial.println("Error: SetMaxFreq requires an integer argument"); return;}
        maxFreq = args[0];

    } else if (strcmp(command,"SetMinFreq") == 0) {
        if (numArgs < 1) {Serial.println("Error: SetMinFreq requires an integer argument"); return;}
        minFreq = args[0];

    } else if (strcmp(command,"SetFrequency") == 0) {
        if (numArgs < 1) {Serial.println("Error: SetFrequency requires an integer argument"); return;}
        sinFrequency = args[0];
    	freqFromPin = false;
        // RH added
        analogWrite(voltageOutputPin, (sinFrequency/10) * (pow(2, ANALOG_WRITE_RES)-1) / MAX_ANALOG_VOLTAGE_MV);
        // analogWrite(voltageOutputPin, (1) * (pow(2, ANALOG_WRITE_RES)-1) / MAX_ANALOG_VOLTAGE_MV);
        ampFromPin = false;
        
    } else if (strcmp(command,"SetAmplitude") == 0) {
        if (numArgs < 1) {Serial.println("Error: SetAmplitude requires an integer argument"); return;}
        sinAmplitude = args[0];
        generateSinTable(sinAmplitude);

    } else if (strcmp(command,"GetFrequencyFromPin") == 0) {
    	freqFromPin = true;

    } else if (strcmp(command,"GetAmplitudeFromPin") == 0) {
    	ampFromPin = true;

    } else {
    	Serial.println("ERROR: Bad Command");
    }
}
