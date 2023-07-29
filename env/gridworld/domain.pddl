(define (domain gridworld)
	(:requirements :strips :typing :equality :negative-preconditions :quantified-preconditions :conditional-effects :domain-axioms :derived-predicates)
	(:types
		movable - object
		openable - object
		lockable - openable
		location - object
	)
	(:constants
		agent - object
		hand - object
	)
	(:predicates
		(reachable ?a - object)
		(closed ?a - object)
		(locked ?a - object)
		(in ?a - object ?b - object)
		(connects ?a - openable ?b - location ?c - location)
		(matches ?a - object ?b - object)
	)

	(:action enter
		; pre: Agent is not in the room and the door that connects the room to
		;      the current room is open.
		; post: Agent is in the room and not in the previous room, room is
		;       reachable (meaning agent can place objects in it), and previous
		;       room is unreachable.
		:parameters (?room - location ?door - openable)
		:precondition (and
			(not (in agent ?room))
			(not (closed ?door))
			(exists (?currentroom - location)
				(and
					(in agent ?currentroom)
					(connects ?door ?room ?currentroom)
				)
			)
		)
		:effect (and
			(in agent ?room)
			(reachable ?door)
			(forall (?obj - object)
				(when
					(not (in ?obj hand))
					(not (reachable ?obj))
				)
			)
			(forall (?currentroom - location)
				(when (connects ?door ?room ?currentroom)
					(not (in agent ?currentroom))
				)
			)
		)
	)
	(:action enter-fail
		; pre: Agent is not in the room and there is no door that connects the
		;      room to the one the agent is in.
		; post: None.
		:parameters (?room - location ?currentroom - location)
		:precondition (and
			(not (in agent ?room))
			(in agent ?currentroom)
			(not (connects ?door ?room ?currentroom))
		)
		:effect (and)
	)

	(:action goto
		; pre: Agent is in the same room as the object.
		; post: Object and all objects inside it become reachable (if it is
		;       open), while all other objects become unreachable.
		:parameters (?obj - object ?room - location)
		:precondition (and
			(not (= agent ?obj))
			(not (reachable ?obj))
			(in agent ?room)
			(in ?obj ?room)
		)
		:effect (and
			(reachable ?obj)
			(forall (?otherobj - object)
				(when
					(and
						(not (= ?otherobj ?obj))
						(not (in ?otherobj hand))
					)
					(not (reachable ?otherobj))
				)
			)
		)
	)

	(:action pick
		; pre: Object is reachable and hand is empty.
		; post: Object is in hand and no longer in its container.
		:parameters (?obj - movable ?container - object)
		:precondition (and
			(or
				(openable ?container)
				(location ?container)
			)
			(in ?obj ?container)
			(or
				(reachable ?obj)
				(reachable_in ?obj)
			)
			(or
				(in agent ?container)
				(reachable ?container)
				(reachable_in ?container)
			)
			(forall
				(?otherobj - movable)
				(not (in ?otherobj hand))
			)
		)
		:effect (and
			(not (in ?obj ?container))
			(in ?obj hand)
		)
	)
	(:action pick-fail
		; pre: Object is reachable and hand is empty.
		; post: Object is in hand and no longer in its container.
		:parameters (?obj - movable ?container - object)
		:precondition (and
			(or
				(openable ?container)
				(location ?container)
			)
			(in ?obj ?container)
			(not (reachable ?obj))
			(not (reachable_in ?obj))
			(not (reachable ?container))
			(not (reachable_in ?container))
			(forall
				(?otherobj - movable)
				(not (in ?otherobj hand))
			)
		)
		:effect (and)
	)

	(:action place
		; pre: Object is in hand and container is reachable and open.
		; post: Object is in the container and not the hand.
		:parameters (?obj - movable ?container - object)
		:precondition (and
			(or
				(openable ?container)
				(location ?container)
			)
			(in ?obj hand)
			(not (= ?obj ?container))
			(not (closed ?container))
			(or
				(in agent ?container)
				(reachable ?container)
				(reachable_in ?container)
			)
		)
		:effect (and
			(not (in ?obj hand))
			(in ?obj ?container)
		)
	)

	(:action open
		; pre: Object is closed, unlocked, and reachable, and hand is empty.
		; post: Object is open and contained objects are reachable.
		:parameters (?obj - openable)
		:precondition (and
			(reachable ?obj)
			(closed ?obj)
			(not (locked ?obj))
			(forall
				(?otherobj - movable)
				(not (in ?otherobj hand))
			)
		)
		:effect (not (closed ?obj))
	)
	(:action open-fail
		; pre: Object is closed and unlocked but not reachable.
		; post: Object is open and contained objects are reachable.
		:parameters (?obj - openable)
		:precondition (and
			(not (reachable ?obj))
			(closed ?obj)
			(not (locked ?obj))
			(forall
				(?otherobj - movable)
				(not (in ?otherobj hand))
			)
		)
		:effect (and)
	)

	(:action close
		; pre: Object is open and reachable and hand is empty.
		; post: Object is closed and contained objects are unreachable.
		:parameters (?obj - openable)
		:precondition (and
			(reachable ?obj)
			(not (closed ?obj))
			(forall
				(?otherobj - movable)
				(not (in ?otherobj hand))
			)
		)
		:effect (and
			(closed ?obj)
			(forall
				(?otherobj - movable)
				(when
					(in ?otherobj ?obj)
					(not (reachable ?otherobj))
				)
			)
		)
	)

	(:action close-fail
		; pre: Object is reachable and unlocked and hand is empty.
		; post: Object is closed and contained objects are unreachable.
		:parameters (?obj - openable)
		:precondition (and
			(not (reachable ?obj))
			(not (closed ?obj))
			(forall
				(?otherobj - movable)
				(not (in ?otherobj hand))
			)
		)
		:effect (and)
	)

	(:action unlock
		; pre: Object is closed and the matching key is in hand.
		; post: Object is unlocked.
		:parameters (?obj - lockable ?key - movable)
		:precondition (and
			(reachable ?obj)
			(locked ?obj)
			(in ?key hand)
			(matches ?key ?obj)
		)
		:effect (not (locked ?obj))
	)
	(:action unlock-fail
		; pre: Object is closed but the key in hand is not matching.
		; post: None.
		:parameters (?obj - lockable ?key - movable)
		:precondition (and
			(reachable ?obj)
			(locked ?obj)
			(in ?key hand)
			(not (matches ?key ?obj))
		)
		:effect (and)
	)

	(:action lock
		; pre: Object is closed and the matching key is in hand.
		; post: Object is locked.
		:parameters (?obj - lockable ?key - movable)
		:precondition (and
			(reachable ?obj)
			(closed ?obj)
			(not (locked ?obj))
			(in ?key hand)
			(matches ?key ?obj)
		)
		:effect (locked ?obj)
	)
	(:action lock-fail
		; pre: Object is closed but the key in hand is not matching.
		; post: None.
		:parameters (?obj - lockable ?key - movable)
		:precondition (and
			(reachable ?obj)
			(closed ?obj)
			(not (locked ?obj))
			(in ?key hand)
			(not (matches ?key ?obj))
		)
		:effect (and)
	)

	(:axiom
		; An object cannot be in itself
		:vars (?a - object ?b - object)
		:context (= ?a ?b)
		:implies (not (in ?a ?b))
	)
	(:axiom
		; Two objects cannot be in hand at the same time
		:vars (?obj - object ?otherobj - object)
		:context (and
			(in ?obj hand)
			(not (= ?obj ?otherobj))
		)
		:implies (not (in ?otherobj hand))
	)
	(:axiom
		; The agent can only be in a location
		:vars (?obj - object)
		:context (in agent ?obj)
		:implies (location ?obj)
	)
	(:axiom
		; An object in hand must be movable
		:vars (?obj - object)
		:context (in ?obj hand)
		:implies (movable ?obj)
	)
	(:axiom
		; An object cannot be in the agent
		:vars (?obj - object ?container - object)
		:context (in ?obj ?container)
		:implies (not (= ?container agent))
	)
	(:axiom
		; A reachable object cannot be a location, hand, or agent
		:vars (?obj - object)
		:context (reachable ?obj)
		:implies (and
			(not (location ?obj))
			(not (= ?obj hand))
			(not (= ?obj agent))
		)
	)
	(:axiom
		; If a reachable object is in a container, the container cannot be closed
		:vars (?obj - object ?container - object)
		:context (and
			(reachable ?obj)
			(in ?obj ?container)
		)
		:implies (not (closed ?container))
	)
	(:axiom
		; If an object is open, it cannot be locked
		:vars (?obj - object)
		:context (not (closed ?obj))
		:implies (not (locked ?obj))
	)
	(:axiom
		; If an object is locked, it must be closed
		:vars (?obj - object)
		:context (locked ?obj)
		:implies (closed ?obj)
	)
	(:axiom
		; If a door connects two locations, it is in both locations
		:vars (?door - openable ?loc_a - location ?loc_b - location)
		:context (connects ?door ?loc_a ?loc_b)
		:implies (and
			(connects ?door ?loc_b ?loc_a)
			(in ?door ?loc_a)
			(in ?door ?loc_b)
		)
	)
	(:axiom
		; If a door connects two locations, it is in both locations
		:vars (?door - openable ?loc_a - location ?loc_b - location)
		:context (not (connects ?door ?loc_a ?loc_b))
		:implies (not (connects ?door ?loc_b ?loc_a))
	)
	(:axiom
		; If an object is in the hand, it is reachable
		:vars (?obj - object)
		:context (in ?obj hand)
		:implies (reachable ?obj)
	)
	(:axiom
		; If an object is reachable, the agent must be in the same location
		:vars (?obj - object ?loc - location)
		:context (and
			(reachable ?obj)
			(in ?obj ?loc)
			(not (= ?obj door))
		)
		:implies (in agent ?loc)
	)
	(:axiom
		; An object cannot be in two containers at once, unless it is a door
		:vars (?obj - object ?container - object)
		:context (and
			(in ?obj ?container)
			(not (= ?obj door))
		)
		:implies (forall (?othercontainer - object)
			(when
				(not (= ?othercontainer ?container))
				(not (in ?obj ?othercontainer))
			)
		)
	)

	(:derived (reachable_in ?obj - object)
		; If an object is reachable, all objects contained inside are reachable
		; unless the container is a room.
		(exists (?container - object)
			(and
				(not (location ?container))
				(in ?obj ?container)
				(or
					(reachable_in ?container)
					(reachable ?container)
				)
				(not (closed ?container))
			)
		)
	)
)
