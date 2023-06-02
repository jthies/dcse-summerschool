// Copyright 2021 Alexander Heinlein
// Contact: Alexander Heinlein (a.heinlein@tudelft.nl)

#ifndef _UTILS
#define _UTILS

// std
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <exception>

// Belos
#include <BelosLinearProblem.hpp>
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

// Teuchos
#include <Teuchos_Array.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_Tuple.hpp>

// Thyra
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROscalar_typeH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>

// Xpetra
#include <Xpetra_Map.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROscalar_typeH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>

// FROSch
#include <ShyLU_DDFROSch_config.h>
#include <FROSch_Tools_def.hpp>
#include <FROSch_SchwarzPreconditioners_fwd.hpp>
#include <FROSch_OneLevelPreconditioner_def.hpp>

// namespaces
using namespace Belos;
using namespace FROSch;
using namespace std;
using namespace Teuchos;
using namespace Xpetra;

// typedefs
typedef MultiVector<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> multivector_type;
typedef multivector_type::scalar_type scalar_type;
typedef multivector_type::local_ordinal_type local_ordinal_type;
typedef multivector_type::global_ordinal_type global_ordinal_type;
typedef multivector_type::node_type node_type;
typedef MultiVectorFactory<scalar_type,local_ordinal_type,global_ordinal_type,node_type> multivectorfactory_type;
typedef Map<local_ordinal_type,global_ordinal_type,node_type> map_type;
typedef Matrix<scalar_type,local_ordinal_type,global_ordinal_type,node_type> matrix_type;
typedef CrsMatrixWrap<scalar_type,local_ordinal_type,global_ordinal_type,node_type> crsmatrixwrap_type;

typedef Galeri::Xpetra::Problem<Map<local_ordinal_type,global_ordinal_type,node_type>,crsmatrixwrap_type,multivector_type> problem_type;

typedef Belos::OperatorT<multivector_type> operatort_type;
typedef Belos::LinearProblem<scalar_type,multivector_type,operatort_type> linear_problem_type;
typedef Belos::SolverFactory<scalar_type,multivector_type,operatort_type> solverfactory_type;
typedef Belos::SolverManager<scalar_type,multivector_type,operatort_type> solver_type;
typedef XpetraOp<scalar_type,local_ordinal_type,global_ordinal_type,node_type> xpetraop_type;

typedef FROSch::OneLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> onelevelpreconditioner_type;
typedef FROSch::TwoLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> twolevelpreconditioner_type;

int assembleSystemMatrix (RCP<const Comm<int> > comm,
                          UnderlyingLib xpetraLib,
                          string equation,
                          int dimension,
                          int N,
                          int M,
                          RCP<matrix_type> &A,
                          RCP<multivector_type> &coordinates)
{
    // Create parameter list for Galeri
    ParameterList GaleriList;
    global_ordinal_type n = N;
    global_ordinal_type m = N*M;
    GaleriList.set("nx",m); GaleriList.set("ny",m); GaleriList.set("nz",m);
    GaleriList.set("mx",n); GaleriList.set("my",n); GaleriList.set("mz",n);

    RCP<const map_type> nodeMap, dofMap;
    RCP<problem_type> problem;
    if (dimension == 2) {
        nodeMap = Galeri::Xpetra::CreateMap<local_ordinal_type,global_ordinal_type,node_type>(xpetraLib,"Cartesian2D",comm,GaleriList);
        coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<scalar_type,local_ordinal_type,global_ordinal_type,map_type,multivector_type>("2D",nodeMap,GaleriList);
        if (!equation.compare("laplace")) {
            dofMap = nodeMap;
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Laplace2D",dofMap,GaleriList);
        } else {
            dofMap = Xpetra::MapFactory<local_ordinal_type,global_ordinal_type,node_type>::Build(nodeMap,2);
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Elasticity2D",dofMap,GaleriList);
        }
        A = problem->BuildMatrix();
    } else if (dimension == 3) {
        nodeMap = Galeri::Xpetra::CreateMap<local_ordinal_type,global_ordinal_type,node_type>(xpetraLib,"Cartesian3D",comm,GaleriList);
        coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<scalar_type,local_ordinal_type,global_ordinal_type,map_type,multivector_type>("3D",nodeMap,GaleriList);
        if (!equation.compare("laplace")) {
            dofMap = nodeMap;
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Laplace3D",dofMap,GaleriList);
        } else {
            dofMap = Xpetra::MapFactory<local_ordinal_type,global_ordinal_type,node_type>::Build(nodeMap,3);
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Elasticity3D",dofMap,GaleriList);
        }
        A = problem->BuildMatrix();
    }
    return 0;
}

#endif
