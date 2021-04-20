% Simple Arduino connection example

function arduinoStepperExample()
	baud = 9600;

	% establish Arduino connection at 9600 baud
	ard = ArduinoConnection(@messageHandler, baud);

	% tell motor for zero itself
	msgToSend = 'Z 1';
	ard.writeString(msgToSend);
	% fprintf('Zeroing motor.\n'); %

	pause(5); % wait a few seconds for zeroing to complete

	% send motor to position <pos>
	pos = 200;
	msgToSend = sprintf('G 1 %i', pos);
	ard.writeString(msgToSend);

	pause(1);

	% send motor to a random sequence of positions
	maxPos = 5000;
	numPos = 20;
	for pos = randi([0, maxPos], 1, numPos)
		msgToSend = sprintf('G 1 %i', pos);
		ard.writeString(msgToSend);
		pause(1);
    end
    
    fclose(ard);
    fprintf('Closing Arduino connection.\n');
end

function messageHandler(msg)
	fprintf('Recived message: %s\n', msg)
end
