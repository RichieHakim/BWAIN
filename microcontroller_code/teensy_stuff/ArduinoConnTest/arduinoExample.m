% Simple Arduino connection example

function arduinoExample()
baud = 9600;
ard = ArduinoConnection(@messageHandler, baud);

% alternative: ard = ArduinoConnection(@(m) disp(m), baud); % this prints the message
% alternative: ard = ArduinoConnection(@(m), baud); % this doesn't


while true
	msgToSend = 'Foo';
	ard.writeString(msgToSend);
	fprintf('Sent message: %s\n', msgToSend)
	pause(5)
end
end

function messageHandler(msg)
	fprintf('Recived message: %s\n', msg)
end
