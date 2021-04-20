%% ArduinoConnetion: serial port connection to an Arduino
classdef ArduinoConnection < handle

properties
	debugMode = false;
	connected = false;
	serialConnection = [];
	arduinoMessageString = '';
	messgeCallbackFcn = [];
	handshakeCharacter = '^';
end

methods
	function obj = ArduinoConnection(messgeCallbackFcn, baudRate)
		obj.messgeCallbackFcn = messgeCallbackFcn;
		obj.arduinoMessageString = '';
		arduinoPortName = obj.findFirstArduinoPort();

		if isempty(arduinoPortName)
		    disp('Can''t find serial port with Arduino')
		    return
		end

		% Define the serial port object.
		fprintf('Starting serial on port: %s\n', arduinoPortName);
		serialPort = serial(arduinoPortName);

		% Set the baud rate
		serialPort.BaudRate = 9600;
		if (nargin >= 2)
			serialPort.BaudRate = baudRate;
		end

		% Add a callback function to be executed whenever 1 line is available
		% to be read from the port's buffer.
		serialPort.BytesAvailableFcn = @(port, event)obj.readMessage(port, event);
		serialPort.BytesAvailableFcnMode = 'terminator';
        serialPort.Terminator = 'CR/LF'; % Ardunio println() commands uses CR/LN termination

		% Open the serial port for reading and writing.
		obj.serialConnection = serialPort;
		fopen(serialPort);

		% wait for Arduino handshake
		% (we write an 'S' to the Arduino and exect an 'S' in return)
% 		fprintf('Waiting for Arduino startup')
%         obj.writeString(obj.handshakeCharacter)
%         waitCounter = 0;
% 		while (~obj.connected)
% 		    fprintf('.');
% 		    pause(0.5);
%             waitCounter = waitCounter + 1;
%             if (mod(waitCounter,10)==0)
%                 obj.writeString(obj.handshakeCharacter)
%             end
% 		end
% 		fprintf('\n')
	end


	function writeMessage(obj, messageChar, arg1, arg2)
	    stringToSend = sprintf('%s %d %d',messageChar, arg1, arg2);
	    obj.writeString(stringToSend);
	end

	function writeString(obj, stringToWrite)
	    fprintf(obj.serialConnection,'%s\n',stringToWrite, 'sync');

	    % DEBUGING
	    if obj.debugMode
	    	disp(['To Arduino: "', stringToSend, '"' ]);
	    end
	end

	function readMessage(obj, port, event)
		% read line from serial buffer
	   	obj.arduinoMessageString = fgetl(obj.serialConnection);
	   	%fprintf('#%s#\n',obj.arduinoMessageString);
	   	% fprintf('--> "');
	   	% [tline,count,msg] = fgetl(obj.serialConnection);
	   	% fprintf('%s" :: %d :: "%s" \n', tline,count,msg);

		% we confirm that the connection was established once the first message is recieved
		if (~obj.connected)
			obj.connected = true;
		end

		% run user code to evaluate the message
        %feval(obj.messgeCallbackFcn, obj.arduinoMessageString);
        obj.messgeCallbackFcn(obj.arduinoMessageString);

	end

	function fclose(obj)
		fclose(obj.serialConnection);
        % delete(obj.serialConnection)
        %clear obj.serialConnection
	end

	function delete(obj)
		obj.fclose();
	end

end

methods (Static)

	function port = findFirstArduinoPort()
		% finds the first port with an Arduino on it.

		serialInfo = instrhwinfo('serial');
		archstr = computer('arch');

		port = [];

		% OSX code:
		if strcmp(archstr,'maci64')
		    for portN = 1:length(serialInfo.AvailableSerialPorts)
		        portName = serialInfo.AvailableSerialPorts{portN};
		        if strfind(portName,'tty.usbmodem')
		            port = portName;
		            return
		        end
		    end
		else
		% PC code:
		    % code from Benjamin Avants on Matlab Answers
		    % http://www.mathworks.com/matlabcentral/answers/110249-how-can-i-identify-com-port-devices-on-windows

		    Skey = 'HKEY_LOCAL_MACHINE\HARDWARE\DEVICEMAP\SERIALCOMM';
		    % Find connected serial devices and clean up the output
		    [~, list] = dos(['REG QUERY ' Skey]);
		    list = strread(list,'%s','delimiter',' ');
		    coms = 0;
		    for i = 1:numel(list)
		      if strcmp(list{i}(1:3),'COM')
		            if ~iscell(coms)
		                coms = list(i);
		            else
		                coms{end+1} = list{i};
		            end
		        end
		    end
		    key = 'HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\USB\';
		    % Find all installed USB devices entries and clean up the output
		    [~, vals] = dos(['REG QUERY ' key ' /s /f "FriendlyName" /t "REG_SZ"']);
		    vals = textscan(vals,'%s','delimiter','\t');
		    vals = cat(1,vals{:});
		    out = 0;
		    % Find all friendly name property entries
		    for i = 1:numel(vals)
		        if strcmp(vals{i}(1:min(12,end)),'FriendlyName')
		            if ~iscell(out)
		                out = vals(i);
		            else
		                out{end+1} = vals{i};
		            end
		        end
		    end
		    % Compare friendly name entries with connected ports and generate output
		    for i = 1:numel(coms)
		        match = strfind(out,[coms{i},')']);
		        ind = 0;
		        for j = 1:numel(match)
		            if ~isempty(match{j})
		                ind = j;
		            end
		        end
		        if ind ~= 0
		            com = str2double(coms{i}(4:end));
		            % Trim the trailing ' (COM##)' from the friendly name - works on ports from 1 to 99
		            if com > 9
		                len = 8;
		            else
		                len = 7;
		            end
		            devs{i,1} = out{ind}(27:end-len);
		            devs{i,2} = coms{i};
		        end
		    end
		    % get the first arduino port
		    for i = 1:numel(coms)
		        [portFriendlyName, portName] = devs{i,:};
		        if strfind(portFriendlyName, 'Arduino')
		            port = portName;
		            return
		        elseif strfind(portFriendlyName, 'Teensy')
		            port = portName;
		            return
		        end
		    end
		end
	end



end
end
