#ifndef PROGRESSION_H
#define PROGRESSION_H
#include <string>
class Progression {
public:
  Progression(const std::string &opName = "", int position = -1,
              int initialValue = 0, int incrementValue = 0,
              int monotonicity = 0)
      : OpName(opName), position(position), initialValue(initialValue),
        incrementValue(incrementValue), monotonicity(monotonicity) {}

  // Copy constructor
  Progression(const Progression &other)
      : OpName(other.OpName), position(other.position),
        initialValue(other.initialValue), incrementValue(other.incrementValue),
        monotonicity(other.monotonicity) {}

  // Destructor
  ~Progression() {}

  // Assignment operator
  Progression &operator=(const Progression &other) {
    if (this != &other) {
      OpName = other.OpName;
      position = other.position;
      initialValue = other.initialValue;
      incrementValue = other.incrementValue;
      monotonicity = other.monotonicity;
    }
    return *this;
  }

  void setOpName(const std::string &name) { OpName = name; }

  void setPosition(int pos) { position = pos; }

  void setInitialValue(int val) { initialValue = val; }

  void setIncrementValue(int val) { incrementValue = val; }

  void setMonotonicity(int mono) { monotonicity = mono; }

  const std::string &getOpName() const { return OpName; }

  int getPosition() const { return position; }

  int getInitialValue() const { return initialValue; }

  int getIncrementValue() const { return incrementValue; }
  
  int getMonotonicity() const { return monotonicity; }

private:
  std::string OpName;
  int position;
  int initialValue;
  int incrementValue;
  int monotonicity;
};
#endif // PROGRESSION_H