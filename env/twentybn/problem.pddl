(define (problem sth-sth-template)
	(:domain sth-sth)
	(:objects
		a - sth
		b - sth
		c - sth
	)
	(:init)
	; Impossible dummy goal just to allow parsing.
	(:goal (and (close a) (far a)))
)
