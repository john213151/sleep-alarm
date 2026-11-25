const int ledPin = 12;    // 졸음 감지 시 켤 LED 핀 번호
const int speakerPin = 8; // 하품 감지 시 울릴 부저 핀 번호

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  pinMode(speakerPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();

    if (signal == 'D') {
      // 🛑 졸음 감지: LED 켜고 부저는 끄기
      digitalWrite(ledPin, HIGH);
      noTone(speakerPin);
    } 
    else if (signal == 'Y') {
      // 🥱 하품 감지: 부저 울리고 LED는 끄기
      digitalWrite(ledPin, LOW);
      tone(speakerPin, 261);
    } 
    else if (signal == '0') {
      // ✅ 정상 상태: 모두 끄기
      digitalWrite(ledPin, LOW);
      noTone(speakerPin);
    }
  }
}
