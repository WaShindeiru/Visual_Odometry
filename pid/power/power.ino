#include <QC2Control.h>

#define BLINK_GPIO 5 // define gpio
unsigned long previousMillis;
unsigned long currentMillis;

//Pin 4 for Data+
//Pin 5 for Data-
//See How to connect in the documentation for more details.
QC2Control quickCharge(3, 2);

void setup() {

  pinMode(BLINK_GPIO, OUTPUT);

  quickCharge.setVoltage(12);

  // delay(1000);
}

void loop() {
  // quickCharge.setVoltage(12);
  quickCharge.setVoltage(12);
  currentMillis = millis();
  if (currentMillis - previousMillis >= 6000) {
      previousMillis = currentMillis;
      digitalWrite(BLINK_GPIO, HIGH); 
      delay(100);
      digitalWrite(BLINK_GPIO, LOW);
  }
}