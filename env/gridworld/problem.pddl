(define (problem get-trophy)
	(:domain gridworld)
	(:objects
		room_a - location
		room_b - location
		door - lockable
		chest - lockable
		door_key - movable
		chest_key - movable
		trophy - movable
	)
	(:init
		(in trophy chest)
		(in chest room_b)
		(in door_key room_a)
		(in chest_key room_a)
		(in door room_a)
		(in door room_b)
		(in agent room_a)
		(closed door)
		(locked door)
		(closed chest)
		(locked chest)
		(matches door_key door)
		(matches chest_key chest)
		(connects door room_a room_b)
		(connects door room_b room_a)
	)
	(:goal (in trophy hand))
)
