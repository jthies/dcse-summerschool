#include "MatrixMarket_Tpetra.hpp"
/*
#include "Tpetra_ComputeGatherMap.hpp"
#include "Tpetra_Details_gathervPrint.hpp"
#include "Tpetra_computeRowAndColumnOneNorms.hpp"
*/
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Core.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <algorithm> // std::transform
#include <cctype> // std::toupper
#include <functional>
#include <memory> // std::unique_ptr
#include <sstream>
#include <tuple>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

namespace {

// initializes/finalizes Tpetra on construction/destruction
class TpetraInstance {
public:
  TpetraInstance (int* argc, char*** argv) {
      Tpetra::initialize (argc, argv);
  }

  ~TpetraInstance () {
      Tpetra::finalize ();
  }
};

// Values of command-line arguments.
struct CmdLineArgs {
  std::string matrixFilename;
  int maxIters = 100;
};

// Read in values of command-line arguments.
bool
getCmdLineArgs (CmdLineArgs& args, int argc, char* argv[])
{
  Teuchos::CommandLineProcessor cmdp (false, true);
  cmdp.setOption ("matrixFilename", &args.matrixFilename, "Name of Matrix "
                  "Market file with the sparse matrix A");


  cmdp.setOption ("maxIters", &args.maxIters,
                  "integer; number of consecutive SpMVs to perform on the same vectors.");

  auto result = cmdp.parse (argc, argv);
  return result == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
}

}//anonymous namespace

int main (int argc, char* argv[])
{

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_INIT;
#endif

  using Teuchos::RCP;
  using std::cerr;
  using std::endl;
  using crs_matrix_type = Tpetra::CrsMatrix<>;
  using MV = Tpetra::MultiVector<>;
  // using mag_type = MV::mag_type;
  using reader_type = Tpetra::MatrixMarket::Reader<crs_matrix_type>;

  TpetraInstance tpetraInstance (&argc, &argv);
  auto comm = Tpetra::getDefaultComm ();

  // Get command-line arguments.
  CmdLineArgs args;
  const bool gotCmdLineArgs = getCmdLineArgs (args, argc, argv);
  if (! gotCmdLineArgs) {
    if (comm->getRank () == 0) {
      cerr << "Failed to get command-line arguments!" << endl;
    }
    return EXIT_FAILURE;
  }

  if (args.matrixFilename == "") {
    if (comm->getRank () == 0) {
      cerr << "Must specify sparse matrix filename!" << endl;
    }
    return EXIT_FAILURE;
  }

  int maxIters = args.maxIters;

  // Read sparse matrix A from Matrix Market file.
  if (comm->getRank() == 0) std::cout << "Read matrix '"<<args.matrixFilename << "'..."<<std::endl;
  RCP<crs_matrix_type> A =
    reader_type::readSparseFile (args.matrixFilename, comm);
  if (A.get () == nullptr) {
    if (comm->getRank () == 0) {
      cerr << "Failed to load sparse matrix A from file "
        "\"" << args.matrixFilename << "\"!" << endl;
    }
    return EXIT_FAILURE;
  }

  RCP<MV> X = Teuchos::rcp (new MV (A->getDomainMap (), 1));
  RCP<MV> Y = Teuchos::rcp (new MV (A->getRangeMap (), 1));
  X->putScalar (1.0);

  if (comm->getRank() == 0) std::cout << "Perform "<< maxIters << " SpMVs..."<<std::endl;

  for (int i=0; i<maxIters; i++)
  {
    A->apply (*X, *Y);
  }


#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif
  return EXIT_SUCCESS;
}
