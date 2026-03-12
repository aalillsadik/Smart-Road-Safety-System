
#include <SoftwareSerial.h>
#include <TinyGPS.h>

int state = 0;
const int pin = 3;
float gpslat, gpslon;

TinyGPS gps;
SoftwareSerial sgps(A3, A2);
SoftwareSerial sgsm(A0, A1);

#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

void sendLocationSMS() 
{
  sgps.listen();
  while (sgps.available()) {
    int c = sgps.read();
    if (gps.encode(c)) {
      gps.f_get_position(&gpslat, &gpslon);
    }
  }

  sgsm.listen();
  sgsm.print("\r");
  delay(1000);
  sgsm.print("AT+CMGF=1\r");
  delay(1000);
  sgsm.print("AT+CMGS=\"+88017169++++\"\r");
  delay(1000);
  sgsm.println("Current Location:");
  sgsm.print("https://www.google.com/maps?q=");
  sgsm.print(gpslat, 6);
  sgsm.print(",");
  sgsm.print(gpslon, 6);
  delay(1000);
  sgsm.write(0x1A);
  delay(1000);
}

void setup() {
  pinMode(pin, INPUT);
  sgsm.begin(9600);
  sgps.begin(9600);
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("ALERT SYSTEM FOR");
  lcd.setCursor(0, 1);
  lcd.print("VEHICLE ACCIDENT");

  sgsm.listen();
  delay(500);
  sgsm.print("AT+CMGF=1\r");
  delay(500);
  sgsm.print("AT+CMGS=\"+8801716+++++\"\r");
  delay(1000);
  sgsm.println("VEHICLE ACCIDENT ALERT PROJECT CONNECTED");
  delay(1000);
  sgsm.write(0x1A);
  delay(1000);
}

void loop() {
  sgps.listen();
  while (sgps.available()) {
    int c = sgps.read();
    if (gps.encode(c)) {
      gps.f_get_position(&gpslat, &gpslon);
    }
  }

  if (digitalRead(pin) == HIGH && state == 0) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("VEHICLE ACCIDENT");
    lcd.setCursor(0, 1);
    lcd.print(" DETECTED");
    sgsm.listen();
    sgsm.print("\r");
    delay(1000);
    sgsm.print("AT+CMGF=1\r");
    delay(1000);
    sgsm.print("AT+CMGS=\"+880171++++\"\r");
    delay(1000);
    sgsm.println("VEHICLE ACCIDENT. LOCATION :");
    sgsm.print("https://www.google.com/maps?q=");
    sgsm.print(gpslat, 6);
    sgsm.print(",");
    sgsm.print(gpslon, 6);
    delay(1000);
    sgsm.write(0x1A);
    delay(1000);
    state = 1;
  }

  // Check for "Location" SMS and send current location
  sgsm.listen();
  while (sgsm.available()) 
{
    if (sgsm.find("Location")) 
{
      sendLocationSMS();
    }
  }

  delay(100);
}

