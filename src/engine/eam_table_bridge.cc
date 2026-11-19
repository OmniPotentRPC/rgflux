// C ABI bridge to rgpot::EAMTablePot for RgpotSurface.
#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/ForceStructs.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

void write_error(char *buf, std::size_t cap, const char *msg) {
  if (buf == nullptr || cap == 0) {
    return;
  }
  std::strncpy(buf, msg, cap - 1);
  buf[cap - 1] = '\0';
}

rgpot::EAMTableFormat to_format(int code) {
  switch (code) {
  case 0:
    return rgpot::EAMTableFormat::Funcfl;
  case 1:
    return rgpot::EAMTableFormat::Setfl;
  case 2:
    return rgpot::EAMTableFormat::FinnisSinclair;
  default:
    throw std::invalid_argument("unknown EAMTableFormat code");
  }
}

} // namespace

extern "C" {

void *rgflux_eam_table_open_file(const char *path, int format, int periodic,
                                 double skin, char *err_buf,
                                 std::size_t err_cap) {
  try {
    rgpot::EAMTableConfig cfg;
    cfg.path = path != nullptr ? path : "";
    cfg.format = to_format(format);
    cfg.periodic = (periodic != 0);
    cfg.stress = false;
    cfg.skin = skin;
    auto pot = std::make_unique<rgpot::EAMTablePot>(cfg);
    return pot.release();
  } catch (const std::exception &ex) {
    write_error(err_buf, err_cap, ex.what());
    return nullptr;
  } catch (...) {
    write_error(err_buf, err_cap, "unknown C++ exception in EAMTablePot");
    return nullptr;
  }
}

void *rgflux_eam_table_open_memory(int z, int nrho, double drho, int nr,
                                   double dr, double cutoff,
                                   const double *frho, const double *rhor,
                                   const double *z2r, int periodic, double skin,
                                   char *err_buf, std::size_t err_cap) {
  try {
    if (nrho <= 0 || nr <= 0 || frho == nullptr || rhor == nullptr ||
        z2r == nullptr) {
      throw std::invalid_argument("invalid EAM table array arguments");
    }
    rgpot::EAMTable table;
    table.nrho = nrho;
    table.drho = drho;
    table.nr = nr;
    table.dr = dr;
    table.cutoff = cutoff;
    table.rhomax = static_cast<double>(nrho - 1) * drho;
    table.z = {z};
    table.frho = {std::vector<double>(frho, frho + nrho)};
    table.rhor = {std::vector<double>(rhor, rhor + nr)};
    table.z2r = {std::vector<double>(z2r, z2r + nr)};
    auto pot = std::make_unique<rgpot::EAMTablePot>(table, periodic != 0, false,
                                                    skin);
    return pot.release();
  } catch (const std::exception &ex) {
    write_error(err_buf, err_cap, ex.what());
    return nullptr;
  } catch (...) {
    write_error(err_buf, err_cap, "unknown C++ exception in EAMTablePot");
    return nullptr;
  }
}

void rgflux_eam_table_close(void *handle) {
  if (handle != nullptr) {
    delete static_cast<rgpot::EAMTablePot *>(handle);
  }
}

int rgflux_eam_table_eval(const void *handle, std::size_t n_atoms,
                          const double *pos, const int *atmnrs,
                          const double *box9, double *energy, double *forces,
                          char *err_buf, std::size_t err_cap) {
  if (handle == nullptr || pos == nullptr || atmnrs == nullptr ||
      box9 == nullptr || energy == nullptr || forces == nullptr) {
    write_error(err_buf, err_cap, "null buffer passed to rgflux_eam_table_eval");
    return 1;
  }
  try {
    const auto *pot = static_cast<const rgpot::EAMTablePot *>(handle);
    rgpot::ForceInput input{
        .nAtoms = n_atoms, .pos = pos, .atmnrs = atmnrs, .box = box9};
    rgpot::ForceOut output{.F = forces,
                           .energy = 0.0,
                           .variance = 0.0,
                           .stress = {},
                           .has_stress = 0};
    pot->forceImpl(input, &output);
    *energy = output.energy;
    return 0;
  } catch (const std::exception &ex) {
    write_error(err_buf, err_cap, ex.what());
    return 2;
  } catch (...) {
    write_error(err_buf, err_cap, "unknown C++ exception in EAMTablePot::forceImpl");
    return 3;
  }
}

} // extern "C"
