// Prints Z, A [g/mol] and mean excitation energy [eV] of NIST elements as Geant4 builds them, and of G4_WATER.
#include "G4NistManager.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include <cstdio>
int main(int argc, char** argv) {
  auto* n = G4NistManager::Instance();
  for (int i = 1; i < argc; ++i) {
    auto* e = n->FindOrBuildElement(argv[i]);
    std::printf("%s %d %.6f %.3f\n", argv[i], (int)e->GetZ(), e->GetA() / (g / mole), e->GetIonisation()->GetMeanExcitationEnergy() / eV);
  }
  auto* w = n->FindOrBuildMaterial("G4_WATER");
  std::printf("G4_WATER I %.3f eV density %.4f\n", w->GetIonisation()->GetMeanExcitationEnergy() / eV, w->GetDensity() / (g / cm3));
}
