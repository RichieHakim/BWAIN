// Simple strand test for Adafruit Dot Star RGB LED strip.
// This is a basic diagnostic tool, NOT a graphics demo...helps confirm
// correct wiring and tests each pixel's ability to display red, green
// and blue and to forward data down the line.  By limiting the number
// and color of LEDs, it's reasonably safe to power a couple meters off
// the Arduino's 5V pin.  DON'T try that with other code!

#include <Adafruit_DotStar.h>
#include <SPI.h>         // COMMENT OUT THIS LINE FOR GEMMA OR TRINKET
#define NUMPIXELS 72 // Number of LEDs in strip
#define BUFFER 1 // Number of blank LED positions on the tail ends
#define SCALE ( NUMPIXELS - 2*BUFFER )/ 1023.0

// Hardware SPI is a little faster, but must be wired to specific pins
// (Arduino Uno = pin 11 for data, 13 for clock, other boards are different).
Adafruit_DotStar strip(NUMPIXELS, DOTSTAR_BRG);
// The last parameter is optional -- this is the color data order of the
// DotStar strip, which has changed over time in different production runs.
// Your code just uses R,G,B colors, the library then reassigns as needed.
// Default is DOTSTAR_BRG, so change this if you have an earlier strip.

void setup() {
  strip.begin(); // Initialize pins for output
  strip.show();  // Turn all LEDs off ASAP
//  Serial.begin(9600);
}

// Runs 10 LEDs at a time along strip, cycling through red, green and blue.
// This requires about 200 mA for all the 'on' pixels + 1 mA per 'off' pixel.

int      head  = 0, tail = 0, width = 2;   // Index of first 'on' and 'off' pixels
uint32_t color = 0x0000ff;      // 'On' color (starts red)


void loop() {
  float sensorVal_position = analogRead(A0)*(5/3.3)*0.95; // takes value between 0-1023 (for 0-5V)
  float sensorVal_brightness = analogRead(A1)*(5/3.3)*0.95 // takes value between 0-1023 (for 0-5V)

  int pos = 0;
  int head = NUMPIXELS - BUFFER - ceil(sensorVal_position * SCALE);
  int tail = head - width;
  
  float brightness = 255*(sensorVal_brightness/1024)  *  0.3;
//  Serial.println(brightness);
  
    for (pos=BUFFER;pos<(NUMPIXELS - BUFFER);pos++)
//    Serial.println(pos);
    {
      if (pos > tail && pos <=  head)
        {
        strip.setPixelColor(pos, 0,0,brightness); // 'On' pixel at head
        }
      else
        {
        strip.setPixelColor(pos, 0); // 'On' pixel at head
        }
    }
  strip.show();                     // Refresh strip
}
